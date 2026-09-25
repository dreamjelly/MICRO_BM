#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "ksd__requests_hit_rate.pct",
    "kvd__requests_hit_rate.pct",
    "l2__requests_hit_rate.pct",
    "llc__requests_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l2__bytes_pipe_l1_mem_global_op_ld.sum",
    "llc__bytes_load_pipe_fabric.sum",
    "llc__transactions_pipe_dram_mem_global_op_ld_lookup_miss.sum",
    "ppu__time_duration.sum",
]


def parse_list(s, typ=int):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def parse_number(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s == "-":
        return None

    # 去掉千分位逗号，支持 "1,234.5"
    s2 = s.replace(",", "")
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s2)
    if not m:
        return None
    return float(m.group(0))

def parse_number(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s == "-":
        return None

    s2 = s.replace(",", "")

    # 只允许整个 cell 是数字，避免把 l2__xxx 解析成 2
    if not re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s2):
        return None

    return float(s2)


def parse_acu_csv(csv_path, metrics):
    result = {m: None for m in metrics}

    # 1. 优先处理“metric 作为列名”的宽表格式
    try:
        df = pd.read_csv(csv_path)
        for m in metrics:
            for col in df.columns:
                col_s = col.strip()
                if col_s == m or col_s.startswith(m + " "):
                    vals = df[col].dropna().tolist()
                    if vals:
                        result[m] = parse_number(vals[0])
                    break
    except Exception:
        pass

    # 2. 再兜底处理 raw 格式：metric 名在某个 cell，value 在后面
    rows = []
    with open(csv_path, "r", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows:
        for i, cell in enumerate(row):
            name = cell.strip()
            for m in metrics:
                if result[m] is not None:
                    continue
                if name == m or name.startswith(m + " "):
                    for x in row[i + 1:]:
                        v = parse_number(x)
                        if v is not None:
                            result[m] = v
                            break

    return result

def run_cmd(cmd, log_path):
    print(" ".join(cmd))
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"[WARN] command failed, see {log_path}")


def plot_metric(df, metric, outdir):
    if metric not in df.columns:
        return

    plt.figure()
    for mode in sorted(df["mode"].unique()):
        sub = df[df["mode"] == mode].sort_values("blocks")
        plt.plot(sub["blocks"], sub[metric], marker="o", label=mode)

    plt.xlabel("blocks")
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    safe = metric.replace("/", "_").replace(".", "_")
    plt.savefig(outdir / f"{safe}.png", dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./cache_scope")
    ap.add_argument("--outdir", default="acu_cache_scope_out")
    ap.add_argument("--blocks", default="1,2,4,8,16,32,64")
    ap.add_argument("--workset-kb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--stride-bytes", type=int, default=64)
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--smem-kb", type=int, default=160)
    ap.add_argument("--modes", default="shared,private")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--acu", default="acu")
    ap.add_argument("--parse-only", action="store_true",
                help="only parse existing csv, do not run acu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blocks_list = parse_list(args.blocks, int)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

    records = []

    for mode in modes:
        for b in blocks_list:
            tag = f"mode_{mode}_blk_{b}_ws_{args.workset_kb}KB"
            csv_path = outdir / f"{tag}.csv"
            rep_path = outdir / f"{tag}"
            log_path = outdir / f"{tag}.log"

            cmd = [
                args.acu,
                "-f",
                "-o", str(rep_path),
                "--metrics", ",".join(metrics),
                "--kernel-name-base", "function",
                "--kernel-name", "cache_scope_kernel",
                "--launch-count", "1",
                "--kill", "no",
                "--page", "raw",
                "--csv-file", str(csv_path),
                args.exe,
                "--mode", mode,
                "--blocks", str(b),
                "--threads", str(args.threads),
                "--workset-kb", str(args.workset_kb),
                "--iters", str(args.iters),
                "--stride-bytes", str(args.stride_bytes),
                "--flush-mb", str(args.flush_mb),
                "--smem-kb", str(args.smem_kb),
            ]

            if args.parse_only:
                if not csv_path.exists():
                    print(f"[WARN] csv not found, skip: {csv_path}")
            else:
                run_cmd(cmd, log_path)

            rec = {
                "mode": mode,
                "blocks": b,
                "workset_kb": args.workset_kb,
                "iters": args.iters,
                "stride_bytes": args.stride_bytes,
                "flush_mb": args.flush_mb,
                "smem_kb": args.smem_kb,
            }

            if csv_path.exists():
                rec.update(parse_acu_csv(csv_path, metrics))
            else:
                for m in metrics:
                    rec[m] = None

            records.append(rec)

    df = pd.DataFrame(records)
    result_csv = outdir / "summary.csv"
    df.to_csv(result_csv, index=False)
    print(f"saved: {result_csv}")

    for m in metrics:
        if m in df.columns:
            plot_metric(df, m, outdir)

    print(f"plots saved under: {outdir}")


if __name__ == "__main__":
    main()
