#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

METRICS = {
    "dram_read_Bps": "dram__bytes_read.sum.per_second",
    "dram_write_Bps": "dram__bytes_write.sum.per_second",
    "dram_total_Bps": "dram__bytes.sum.per_second",

    "dram_read_pct": "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram_write_pct": "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram_total_pct": "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    "ppu_dram_pct": "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed",

    "llc_hit_pct": "llc__requests_hit_rate.pct",
    "l2_hit_pct": "l2__requests_hit_rate.pct",

    "occupancy_warps_per_cu": "launch__occupancy_warps_per_cu",
    "waves_per_cu": "launch__waves_per_cu",
    "cu_count": "device__attribute_cu_count",
    "max_warps_per_cu": "device__attribute_max_warps_per_cu",
    "tfu_active_num": "device__attribute_tfu_in_cu_active_num",
}

CASE_RE = re.compile(
    r"dram_(?P<mode>[^_]+)"
    r"_core(?P<active_cores>\d+)"
    r"_blk(?P<blocks>\d+)"
    r"_thr(?P<threads>\d+)"
    r"_warp(?P<total_warps>\d+)"
    r"_gib(?P<gib>[0-9.]+)"
    r"_it(?P<iters>\d+)"
)

MODE_ORDER = {"read": 0, "copy": 1, "write": 2}


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"n/a", "nan", "none", "-"}:
        return None
    s = s.replace(",", "").replace("%", "")
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def read_text(path: Path):
    for enc in ("utf-8", "gbk", "latin1"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            pass
    return ""


def read_csv_rows(path: Path):
    text = read_text(path)
    rows = []
    for row in csv.reader(text.splitlines()):
        rows.append(row)
    return rows


def parse_case_from_name(name: str):
    m = CASE_RE.search(name)
    if not m:
        return {}
    d = m.groupdict()
    for k in ["active_cores", "blocks", "threads", "total_warps", "iters"]:
        d[k] = int(d[k])
    d["gib"] = float(d["gib"])
    return d


def parse_log(log_path: Path):
    out = {}
    if not log_path or not log_path.exists():
        return out

    text = read_text(log_path)

    m = re.search(
        r"mode=(\w+),\s*array=([0-9.]+)\s*GiB,\s*iters=(\d+),\s*blocks=(\d+),\s*threads=(\d+)",
        text,
    )
    if m:
        out["log_mode"] = m.group(1)
        out["log_array_gib"] = float(m.group(2))
        out["log_iters"] = int(m.group(3))
        out["log_blocks"] = int(m.group(4))
        out["log_threads"] = int(m.group(5))

    m = re.search(
        r"kernel_time=([0-9.]+)\s*ms,\s*requested_user_BW=([0-9.]+)\s*GB/s",
        text,
    )
    if m:
        out["kernel_time_ms"] = float(m.group(1))
        out["requested_user_GBps"] = float(m.group(2))

    m = re.search(r"checksum=([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)", text)
    if m:
        out["checksum"] = " ".join(m.groups())

    if "invalid configuration argument" in text:
        out["error"] = "invalid configuration argument"
    elif "No kernels were profiled" in text:
        out["error"] = "No kernels were profiled"
    else:
        out["error"] = ""

    return out


def find_metric_in_wide_table(rows, metric):
    """
    兼容这种格式：
    header: ..., dram__bytes_read.sum.per_second, ...
    data:   ..., 2238..., ...
    """
    for i, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            if metric == cell.strip() or metric in cell:
                for data_row in rows[i + 1:]:
                    if col_idx < len(data_row):
                        v = to_float(data_row[col_idx])
                        if v is not None:
                            return v
    return None


def find_metric_in_raw_rows(rows, metric):
    """
    兼容这种格式：
    ..., Metric Name, Metric Value, ...
    ..., dram__bytes_read.sum.per_second, 2238..., ...
    """
    candidates = []

    for row in rows:
        for idx, cell in enumerate(row):
            c = cell.strip()
            if c == metric or metric in c:
                nums_after = []
                for x in row[idx + 1:]:
                    v = to_float(x)
                    if v is not None:
                        nums_after.append(v)

                if nums_after:
                    candidates.append(nums_after[-1])
                else:
                    nums_all = [to_float(x) for x in row]
                    nums_all = [x for x in nums_all if x is not None]
                    if nums_all:
                        candidates.append(nums_all[-1])

    return candidates[0] if candidates else None


def parse_acu_csv(csv_path: Path):
    rows = read_csv_rows(csv_path)
    out = {}

    for short_name, metric in METRICS.items():
        v = find_metric_in_wide_table(rows, metric)
        if v is None:
            v = find_metric_in_raw_rows(rows, metric)
        out[short_name] = v

    # 派生 TB/s
    for src, dst in [
        ("dram_read_Bps", "dram_read_TBps"),
        ("dram_write_Bps", "dram_write_TBps"),
        ("dram_total_Bps", "dram_total_TBps"),
    ]:
        v = out.get(src)
        out[dst] = v / 1e12 if v is not None else None

    return out


def find_log_for_csv(csv_path: Path):
    same = csv_path.with_suffix(".log")
    if same.exists():
        return same

    logs = list(csv_path.parent.glob("*.log"))
    if logs:
        return logs[0]

    return None


def collect(outdir: Path):
    rows = []

    for csv_path in sorted(outdir.rglob("*.csv")):
        case_name = csv_path.stem
        case_info = parse_case_from_name(case_name)

        # 有些 csv 名可能不是 case 名，用父目录再试一次
        if not case_info:
            case_name = csv_path.parent.name
            case_info = parse_case_from_name(case_name)

        if not case_info:
            continue

        log_path = find_log_for_csv(csv_path)

        row = {}
        row.update(case_info)
        row.update(parse_log(log_path))
        row.update(parse_acu_csv(csv_path))

        row["csv_path"] = str(csv_path)
        row["log_path"] = str(log_path) if log_path else ""

        rows.append(row)

    rows.sort(key=lambda r: (
        MODE_ORDER.get(r.get("mode", ""), 99),
        r.get("active_cores", 0),
        r.get("blocks", 0),
    ))

    return rows


def write_summary_csv(rows, output: Path):
    cols = [
        "mode",
        "active_cores",
        "blocks",
        "threads",
        "total_warps",
        "gib",
        "iters",

        "kernel_time_ms",
        "requested_user_GBps",

        "dram_read_TBps",
        "dram_write_TBps",
        "dram_total_TBps",

        "dram_read_pct",
        "dram_write_pct",
        "dram_total_pct",
        "ppu_dram_pct",

        "llc_hit_pct",
        "l2_hit_pct",

        "occupancy_warps_per_cu",
        "waves_per_cu",
        "cu_count",
        "max_warps_per_cu",
        "tfu_active_num",

        "error",
        "csv_path",
        "log_path",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows, output: Path):
    cols = [
        "mode",
        "active_cores",
        "blocks",
        "threads",
        "total_warps",
        "kernel_time_ms",
        "requested_user_GBps",
        "dram_read_TBps",
        "dram_write_TBps",
        "dram_total_TBps",
        "dram_read_pct",
        "dram_write_pct",
        "ppu_dram_pct",
        "occupancy_warps_per_cu",
        "waves_per_cu",
    ]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    with output.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", help="acu_dram_core_ramp_full_gib1 等结果目录")
    ap.add_argument("-o", "--output", default="dram_core_ramp_summary.csv")
    ap.add_argument("--md", default="dram_core_ramp_summary.md")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    rows = collect(outdir)

    write_summary_csv(rows, Path(args.output))
    write_markdown(rows, Path(args.md))

    print(f"cases: {len(rows)}")
    print(f"csv: {args.output}")
    print(f"md : {args.md}")


if __name__ == "__main__":
    main()
