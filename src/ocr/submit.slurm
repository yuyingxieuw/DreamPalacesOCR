#!/bin/bash
#SBATCH --job-name=ocr8
#SBATCH --account=*** 
#SBATCH --qos=normal
#SBATCH --gpus=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=4:30:00
#SBATCH --output=/gpfs/scrubbed/yxie993/ocr_runs/ocr_out_v2/slurm_%j.log
#SBATCH --error=/gpfs/scrubbed/yxie993/ocr_runs/ocr_out_v2/slurm_%j.log

# ============ premeter ============
MANIFEST="/gpfs/scrubbed/yxie993/manifest.jsonl"   
SCRIPT="/gpfs/scrubbed/yxie993/bench_prefetch.py"
OUT_DIR="/gpfs/scrubbed/yxie993/ocr_runs/ocr_out_v2"
BATCH=8
RENDER_WORKERS=1                
NUM_SHARDS=8
DPI=200
# ==================================

mkdir -p "$OUT_DIR"

# ---- limit cpu/gpu core ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DETECTOR_POSTPROCESSING_CPU_WORKERS=3

# ---- activate surya environment ----
source ~/.bashrc
conda activate /gpfs/scrubbed/yxie993/envs/surya

echo "==== JOB $SLURM_JOB_ID on $(hostname) ===="
echo "manifest=$MANIFEST dpi=$DPI batch=$BATCH shards=$NUM_SHARDS out=$OUT_DIR"
echo "start: $(date)"

# ---- record GPU usage rate----
nvidia-smi dmon -s u -o T > "$OUT_DIR/gpu_util_${SLURM_JOB_ID}.log" 2>&1 &
DMON_PID=$!

# ----  8 shard，each for one gpu ----
pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i \
    python "$SCRIPT" \
        --manifest "$MANIFEST" \
        --out-dir "$OUT_DIR" \
        --shard-index "$i" \
        --num-shards "$NUM_SHARDS" \
        --batch "$BATCH" \
        --render-workers "$RENDER_WORKERS" \
        --dpi "$DPI" \
        > "$OUT_DIR/shard_${i}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> GPU $i (pid ${pids[-1]})"
done

# ---- wait for all shard finish ----
for pid in "${pids[@]}"; do
    wait "$pid"
done

# ---- stop GPU monitor ----
kill $DMON_PID 2>/dev/null

echo "end: $(date)"
echo "==== all shards finished ===="

# ---- 汇总每个 shard 的最后一行（吞吐）----
echo "---- per-shard summary ----"
for i in $(seq 0 $((NUM_SHARDS-1))); do
    echo "shard $i: $(grep FINISHED "$OUT_DIR/shard_${i}_${SLURM_JOB_ID}.log" | tail -1)"
done

# ---- 统计当前累计完成数---- 
echo "total completed so far:"
cat "$OUT_DIR"/shard_*.results.jsonl 2>/dev/null | \
  python -c "import sys,json;ids=set();[ids.add(json.loads(l)['id']) for l in sys.stdin if l.strip()];print(len(ids))"
