# 带宽测试复现命令清单

> 目标：重新跑代码，复现 `bw_ret.xlsx` 中 4 个 Sheet 的结果数据。
> 环境：PPU（sm_80a），`acu` 为 Nsight Compute 封装命令。

---

## Sheet 1(a)：L2/LLC 共享架构测试

**目标**：blocks=1..20，shared 模式，观察 L2/LLC hit 拐点（5/9/13/17 → 4 CU 共享 1 个 L2，20 CU 共享 1 个 LLC）

**代码**：`a_test_l2/cache_scope.cu`

```bash
cd a_test_l2
nvcc -O3 -lineinfo cache_scope.cu -o cache_scope
```

**运行**：

```bash
# blocks=1..20, shared 模式
# test.sh 原始写的是 1..128 + workset 4096KB，但 4096KB > L2(1MB) 会导致拐点不明显
# 建议用 512KB workset 复现 Excel 的拐点 pattern；若不明显再试 256KB / 128KB
python3 run_acu_cache_scope.py \
  --exe ./cache_scope \
  --outdir l2_cluster_shared_512KB \
  --modes shared \
  --blocks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --workset-kb 512 \
  --iters 1 \
  --flush-mb 512 \
  --smem-kb 160
```

**关键参数**：

- `--smem-kb 160`：每 block 占 160KB shared memory，强制每 CU 只驻留 1 个 block，确保 block→CU 映射可预测
- `--iters 1`：只读 1 轮，第 1 个 block cold miss，同 L2 cluster 内后续 block hit
- `--flush-mb 512`：正式测试前 flush cache

**提取**：从 `l2_cluster_shared_512KB/summary.csv` 取 `l2__requests_hit_rate.pct` 和 `llc__requests_hit_rate.pct`，对应 Excel 的 `L2 hit` / `LLC hit` 列。

---

## Sheet 1(b)：KVD/L2/LLC 容量测试

**目标**：blocks=4（同一 L2 cluster），private 模式，扫 workset 64KB..32MB，iters=100，判断 L2≈1MB / LLC≈64MB

**代码**：同 `a_test_l2/cache_scope.cu`

**运行**：遍历 workset 列表（Excel 中的 26 个点）：

```bash
for ws in 64 128 256 288 320 352 384 416 448 480 512 1024 2048 4096 8192 16384 \
         17408 18432 20480 22528 24576 26624 28672 30720 31680 32768; do
  python3 run_acu_cache_scope.py \
    --exe ./cache_scope \
    --outdir capacity_private_ws${ws}KB \
    --modes private \
    --blocks 4 \
    --workset-kb $ws \
    --iters 100 \
    --flush-mb 512 \
    --smem-kb 160
done
```

**关键参数**：

- `--blocks 4`：4 个 block 落在同一个 L2 cluster（4 CU 共享 1 L2）
- `--mode private`：每个 block 读独立工作集，cluster 总工作集 = 4 × workset_per_block
- `--iters 100`：反复读，观察 working set 能否被 L2 保留

**提取**：每个 workset 取 `l2__requests_hit_rate.pct` 和 `llc__requests_hit_rate.pct`。Excel 中 `cluster 总工作集` = 4 × workset_per_block。L2 hit 在 256KB/block（1MB cluster）时 ≈ 0.99，288KB/block（1.12MB cluster）时骤降到 0.57 → L2 容量 ≈ 1MB。LLC hit 在 16MB/block（64MB cluster）时 ≈ 0.99，17MB/block 时开始下降 → LLC 容量 ≈ 64MB。

---

## Sheet 2：各级存储峰值带宽测试

### 2.1 DRAM 读峰值（2.77 TB/s）

**代码**：`a_test_bw/dram_bw.cu`

```bash
cd a_test_bw
nvcc -O3 -arch=sm_80a dram_bw.cu -o dram_bw

# 16 GiB 工作集, 20 iters, read, 1024 blocks, 默认 256 threads
acu --profile-from-start off \
  --metrics="regex:^dram.*$,regex:^ppu__dram.*$,regex:^kvd__.*$,regex:^l2__bytes.*$,regex:^llc__bytes.*$" \
  --launch-count 1 --kill no \
  -o dram_read \
  ./dram_bw 16 20 read 1024
```

**提取**：

- `dram__bytes_read.sum.per_second` → 2.238 TB/s
- `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` → 80.81%
- 理论峰值 = 2.238 / 0.8081 ≈ 2.77 TB/s

### 2.2 LLC 读峰值（6.96 TB/s）

**代码**：`a_test_bw/llc_bw.cu`

```bash
cd a_test_bw
nvcc -O3 -arch=sm_80a llc_bw.cu -o llc_bw

# 48 MiB 工作集 (< 64MB LLC), 5000 iters, 1024 blocks
acu --profile-from-start off \
  --metrics="regex:^llc.*$,regex:^l2__bytes.*$,regex:^kvd__bytes.*$,regex:^kvd__transactions.*$,regex:^dram.*$" \
  --launch-count 1 --kill no \
  -o llc_read \
  ./llc_bw 48 5000 1024
```

**提取**：

- `llc__bytes_load_pipe_fabric.sum.per_second` → 3.146 TB/s
- `llc__throughput.avg.pct_of_peak_sustained_elapsed` → 45.18%
- 理论峰值 = 3.146 / 0.4518 ≈ 6.96 TB/s

### 2.3 L2 读峰值（13.93 TB/s）

**代码**：`a_test_bw/l2_bw.cu`

```bash
cd a_test_bw
nvcc -O3 -arch=sm_80a l2_bw.cu -o l2_bw

acu -f --profile-from-start off \
  --metrics="regex:^l2.*$,regex:^kvd__bytes.*$,regex:^kvd__transactions.*$,regex:^llc.*$,regex:^dram.*$,regex:^launch.*$" \
  --launch-count 1 --kill no \
  --page raw --csv-file l2_bw_192k.csv \
  -o l2_bw_192k \
  ./l2_bw --blocks 64 --threads 512 --workset-kb 192 --iters 5000 --smem-kb 160
```

**提取**：

- `l2__bytes_load_pipe_l1.sum.per_second` → 1.803 TB/s
- `l2__throughput.avg.pct_of_peak_sustained_elapsed` → 12.95%
- 理论峰值 = 1.803 / 0.1295 ≈ 13.93 TB/s

---

## Sheet 3：dram 读写带宽（爬坡）

**目标**：基础 `dram_bw`，read + write，threads=1024，gib=1，iters=20，blocks 扫 40 个点

**代码**：`a_test_bw2/dram_bw.cu`（与 a_test_bw 相同）

```bash
cd a_test_bw2
nvcc -O3 -arch=sm_80a dram_bw.cu -o dram_bw
```

**运行**（用 `run_dram_core_ramp_full.sh`，注意 MODES 改为 "read write"）：

```bash
OUTDIR=acu_dram_core_ramp_full \
EXE=./dram_bw \
GIB=1 \
ITERS=20 \
THREADS=1024 \
WARPS_PER_CORE=64 \
MODES="read write" \
./run_dram_core_ramp_full.sh
```

**参数推导**：

- THREADS=1024 → warps_per_block = 32
- WARPS_PER_CORE=64 → blocks_per_core = 2
- blocks = active_cores × 2
- CORE_LIST 默认 40 个值 → blocks = 2,4,6,...,128,136,144,...,256（与 Excel 完全一致）

**解析**：

```bash
python3 extract_dram_core_ramp.py acu_dram_core_ramp_full \
  -o dram_core_ramp_summary.csv \
  --md dram_core_ramp_summary.md
```

**提取**：`dram_core_ramp_summary.csv` 中的列对应 Excel Sheet 3：

- `dram_read_TBps` / `dram_write_TBps` / `dram_total_TBps`
- `dram_read_pct` / `dram_write_pct` / `ppu_dram_pct`
- `llc_hit_pct` / `l2_hit_pct`
- `occupancy_warps_per_cu` / `cu_count` / `max_warps_per_cu` / `tfu_active_num`

**注意**：解析脚本会输出 `active_cores` 列，Excel 里没有这列（readme.txt 说"不用管 active_cores，没有什么意义"），导入 Excel 时忽略即可。

---

## Sheet 4：LLC fabric 带宽测试

**目标**：9 个 case，固定 smid 组合，load-only，4MB shared working set，1024 threads，1 轮，flush cache

**代码**：`test_llc_crossbar/llc_bw_smid_mask.cu`

```bash
cd test_llc_crossbar
nvcc -O3 -lineinfo -arch=sm_80a llc_bw_smid_mask.cu -o llc_bw_smid_mask
```

**运行**（直接用 `run.sh`，已包含全部 9 个 case）：

```bash
mkdir -p acu_llc_ce_shared_ws4096
./run.sh
```

`run.sh` 会依次跑 A1/A2/A3、B1/B2/B3/B4、C1/C2，每个 case 的参数：

- `--addr-mode shared`（所有 smid 读同一份 4MB 数据）
- `--workset-kb 4096`（4MB）
- `--threads 1024`
- `--blocks-per-smid 1`（每个 smid 只保留 1 个 active block）
- `--grid 4096`（发射足够多的 block 以覆盖所有目标 smid）
- `--iters 1`（1 轮）
- `--flush-mb 512`（flush cache）
- `--prime-repeats 0`（不预先 prime 数据进 LLC）

**提取**：每个 case 的 CSV 中取：

- `llc__bytes_load_pipe_fabric.sum.per_second` → Excel 的 `llc__bytes_load_pipe_fabric.sum.per_second [GB/second]`
- `dram__bytes.sum.per_second` → Excel 的 `dram__bytes.sum.per_second [GB/second]`
- `l2__requests_hit_rate.pct` → Excel 的 `L2 hit`
- `llc__requests_hit_rate.pct` → Excel 的 `LLC hit`

**预期结果**：

- A 组（同 group）：LLC fabric BW ≈ 12.5 GB/s（1 group 不跨 LLC）
- B1/B2/C1/C2：LLC fabric BW 随 group 数线性增长（25/52/105/211 GB/s）
- A 组 L2 hit：A1≈2%, A2≈50%, A3≈75%（同组内 L2 复用）

---

## 验证注意点

1. **Sheet 1(a) 的 workset-kb**：`test.sh` 写的是 4096，但 4MB > L2(1MB) 时 (b-1)/b 拐点 pattern 不成立。建议先用 512KB 跑，看 L2 hit 是否出现 0 → 0.5 → 0.667 → 0.75 的 pattern。若不对，再试 256KB / 128KB。

2. **Sheet 4 的 `--prime-repeats`**：`run.sh` 里设为 0（不 prime），代码默认是 1。Excel A 组 L2 hit 数据（A1≈2%）符合不 prime 的预期（cold start）。保持 0 即可。

---

## 代码与 Sheet 对应总表

| Excel Sheet   | 代码                            | 文件夹            | 状态  |
| ------------- | ------------------------------- | ----------------- | ----- |
| 1(a) 共享架构 | `cache_scope.cu --mode shared`  | a_test_l2         | ready |
| 1(b) 容量     | `cache_scope.cu --mode private` | a_test_l2         | ready |
| 2 DRAM 峰值   | `dram_bw.cu`                    | a_test_bw         | ready |
| 2 LLC 峰值    | `llc_bw.cu`                     | a_test_bw         | ready |
| 2 L2 峰值     | `l2_bw.cu`                      | a_test_bw         | ready |
| 3 读写爬坡    | `dram_bw.cu`（基础版）          | a_test_bw2        | ready |
| 4 LLC fabric  | `llc_bw_smid_mask.cu`           | test_llc_crossbar | ready |
