
#!/usr/bin/env bash
# Launch 8 workers on the current node, one per GPU.
# Each worker sees exactly one GPU (as cuda:0) because CUDA_VISIBLE_DEVICES is
# set per-process before Python starts. Use this for an interactive session /
# salloc; use submit.slurm for a batch job.
set -euo pipefail
 
MANIFEST="${MANIFEST:-manifest.jsonl}"
OUT_DIR="${OUT_DIR:-ocr_out}"
NUM_SHARDS="${NUM_SHARDS:-8}"
 
mkdir -p "$OUT_DIR" logs
 
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$i" \
    python run_shard.py \
        --manifest "$MANIFEST" \
        --out-dir "$OUT_DIR" \
        --shard-index "$i" \
        --num-shards "$NUM_SHARDS" \
        > "logs/shard_${i}.log" 2>&1 &
    pids+=($!)
    echo "started shard $i on GPU $i (pid ${pids[-1]})"
done
 
# Wait for all; exit non-zero if any shard failed.
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
exit $fail
 
