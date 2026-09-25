#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-./dram_bw}
OUTDIR=${OUTDIR:-acu_dram_core_ramp_full}

GIB=${GIB:-16}
ITERS=${ITERS:-20}

# 单 block 最大用 1024 threads = 32 warps
THREADS=${THREADS:-1024}

# 每个“目标核”尝试给满 64 resident warps
WARPS_PER_CORE=${WARPS_PER_CORE:-64}

CORE_LIST=${CORE_LIST:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 24 28 32 40 48 56 64 68 72 76 80 84 88 92 96 100 108 116 124 128"}
# CORE_LIST=${CORE_LIST:-"68 72 76 80 84 88 92 96 100 108 116 124 128"}
MODES=${MODES:-"read copy write"}

mkdir -p "${OUTDIR}"

if (( THREADS % 32 != 0 )); then
  echo "THREADS must be multiple of 32"
  exit 1
fi

warps_per_block=$((THREADS / 32))

if (( WARPS_PER_CORE % warps_per_block != 0 )); then
  echo "WARPS_PER_CORE=${WARPS_PER_CORE} must be divisible by warps_per_block=${warps_per_block}"
  exit 1
fi

blocks_per_core=$((WARPS_PER_CORE / warps_per_block))

echo "EXE=${EXE}"
echo "OUTDIR=${OUTDIR}"
echo "GIB=${GIB}, ITERS=${ITERS}"
echo "THREADS=${THREADS}, warps_per_block=${warps_per_block}"
echo "WARPS_PER_CORE=${WARPS_PER_CORE}, blocks_per_core=${blocks_per_core}"
echo "CORE_LIST=${CORE_LIST}"
echo "MODES=${MODES}"

for mode in ${MODES}; do
  for active_cores in ${CORE_LIST}; do

    blocks=$((active_cores * blocks_per_core))
    total_warps=$((blocks * warps_per_block))

    name=dram_${mode}_core${active_cores}_blk${blocks}_thr${THREADS}_warp${total_warps}_gib${GIB}_it${ITERS}
    casedir=${OUTDIR}/${mode}/${name}
    mkdir -p "${casedir}"

    echo "==== ${name} ===="
    echo "mode=${mode}, active_cores=${active_cores}, blocks=${blocks}, threads=${THREADS}, warps_per_block=${warps_per_block}, total_warps=${total_warps}"

    acu \
      --profile-from-start off \
      --launch-count 1 \
      --kill no \
      --page raw \
      --csv-file "${casedir}/${name}.csv" \
      --set full \
      --replay-mode kernel \
      -o "${casedir}/${name}" \
      -f \
      "${EXE}" "${GIB}" "${ITERS}" "${mode}" "${blocks}" "${THREADS}" \
      2>&1 | tee "${casedir}/${name}.log"

  done
done
