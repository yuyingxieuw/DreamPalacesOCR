#!/bin/bash
#SBATCH --job-name=ocr8
#SBATCH --account=dreampalaces
#SBATCH --qos=normal
#SBATCH --gpus=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/yxie993/ocr_runs/%j/slurm.log
#SBATCH --error=/gpfs/scrubbed/yxie993/ocr_runs/%j/slurm.log

# ============ 可调参数 ============
MANIFEST="/gpfs/scrubbed/yxie993/small.jsonl"   
SCRIPT="/gpfs/scrubbed/yxie993/bench_prefetch.py"
BATCH=8
RENDER_WORKERS=1                
NUM_SHARDS=8
# 输出目录用 job id，每次跑独立，不互相覆盖
OUT_DIR="/gpfs/scrubbed/yxie993/ocr_runs/${SLURM_JOB_ID}"
# ==================================

mkdir -p "$OUT_DIR"

# ---- 限核：防止 8 个 shard 互相抢核 ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DETECTOR_POSTPROCESSING_CPU_WORKERS=3

# ---- 激活环境 ----
source ~/.bashrc
conda activate /gpfs/scrubbed/yxie993/envs/surya

echo "==== JOB $SLURM_JOB_ID on $(hostname) ===="
echo "manifest=$MANIFEST batch=$BATCH shards=$NUM_SHARDS render_workers=$RENDER_WORKERS"
echo "nproc=$(nproc)  cpu_count(py)=$(python -c 'import os;print(os.cpu_count())')"
echo "out=$OUT_DIR"
echo "start: $(date)"

# ---- 后台记录 GPU 利用率（全程，带时间戳）----
nvidia-smi dmon -s u -o T > "$OUT_DIR/gpu_util.log" 2>&1 &
DMON_PID=$!
echo "gpu monitor pid=$DMON_PID -> $OUT_DIR/gpu_util.log"

# ---- 起 8 个 shard，各绑一卡 ----
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
        > "$OUT_DIR/shard_${i}.log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> GPU $i (pid ${pids[-1]})"
done

# ---- 等所有 shard 结束 ----
for pid in "${pids[@]}"; do
    wait "$pid"
done

# ---- 停止 GPU 监控 ----
kill $DMON_PID 2>/dev/null

echo "end: $(date)"
echo "==== all shards finished ===="

# ---- 汇总每个 shard 的最后一行（吞吐）----
echo "---- per-shard summary ----"
for i in $(seq 0 $((NUM_SHARDS-1))); do
    echo "shard $i: $(grep FINISHED "$OUT_DIR/shard_${i}.log" | tail -1)"
done
