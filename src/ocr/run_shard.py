"""
bench batch=8 + prefetch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
python bench_prefetch.py --manifest manifest.jsonl --out-dir /tmp/test_pf \
    --shard-index 0 --num-shards 1000 --batch 8 --render-workers 1
"""

## batch=8 + prefetch: Pre-render next batch in background to overlap with current batch GPU inference
import argparse, json, os, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PDF_DIR = Path("/gpfs/scrubbed/yxie993/pdfs")

def load_shard(manifest, shard_index, num_shards):
    jobs = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            if i % num_shards == shard_index:
                jobs.append(json.loads(line))
    return jobs

def load_completed_ids(results_path):
    if not results_path.exists():
        return set()
    ids = set()
    with results_path.open() as f:
        for line in f:
            if not line.endswith("\n"):
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = obj.get("id")
            if jid:
                ids.add(jid)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--shard-index", required=True, type=int)
    ap.add_argument("--num-shards", required=True, type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--render-workers", type=int, default=1)  
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[shard {args.shard_index}/{args.num_shards}]"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if visible == "<unset>":
        raise SystemExit(f"{tag} ERROR: CUDA_VISIBLE_DEVICES not set")
    if "," in visible:
        raise SystemExit(f"{tag} ERROR: sees multiple GPUs ({visible})")
    print(f"{tag} GPU={visible} cpu_count={os.cpu_count()} "
          f"batch={args.batch} render_workers={args.render_workers}", flush=True)

    results_path = args.out_dir / f"shard_{args.shard_index}.results.jsonl"
    fail_log = args.out_dir / f"shard_{args.shard_index}.failed.log"
    completed = load_completed_ids(results_path)

    from surya.input.processing import open_pdf, get_page_images
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    jobs = load_shard(args.manifest, args.shard_index, args.num_shards)
    jobs = [j for j in jobs if j["file_id"] not in completed]
    print(f"{tag} {len(completed)} done, {len(jobs)} to do", flush=True)

    def render_one(job):
        """render single page; success return (job, img), fail return (job, None)。"""
        try:
            path = PDF_DIR / f"{job['file_id']}.pdf"
            doc = open_pdf(str(path))
            img = get_page_images(doc, [0], dpi=args.dpi)[0]
            doc.close()
            return job, img
        except Exception:
            return job, None

    # cut jobs into batches
    batches = [jobs[i:i+args.batch] for i in range(0, len(jobs), args.batch)]

    done = failed = 0
    t_start = time.time()

    def render_batch(pool, batch_jobs):
        """multithreading render batch return future list。"""
        return [pool.submit(render_one, j) for j in batch_jobs]

    with results_path.open("a") as out_f, fail_log.open("a") as fail_f, \
         ThreadPoolExecutor(max_workers=args.render_workers) as pool:

        if not batches:
            print(f"{tag} nothing to do", flush=True)
            return

        # start the first batch
        pending_futs = render_batch(pool, batches[0])

        for bi in range(len(batches)):
            # 1) collect result
            imgs, ok_jobs = [], []
            for fut in pending_futs:
                job, img = fut.result()
                if img is None:
                    fail_f.write(f"{job['file_id']}\tRENDER_FAIL\n")
                    failed += 1
                else:
                    imgs.append(img)
                    ok_jobs.append(job)
            fail_f.flush()

            # 2) Immediately trigger background rendering for the next batch so the CPU
            #    renders it in parallel while the GPU processes the current batch (overlapping).
            if bi + 1 < len(batches):
                pending_futs = render_batch(pool, batches[bi + 1])

            if not imgs:
                continue

            # 3) GPU processes the current batch (while the next batch is being rendered in the background).
            try:
                preds = recognition(imgs, det_predictor=detection)
            except Exception:
                err = traceback.format_exc().strip().replace("\n", " | ")
                for job in ok_jobs:
                    fail_f.write(f"{job['file_id']}\tBATCH_FAIL {err}\n")
                    failed += 1
                fail_f.flush()
                continue

            # 4) return back
            for job, pred in zip(ok_jobs, preds):
                jid = job["file_id"]
                text_lines = [line.text for line in pred.text_lines]
                record = {
                    "id": jid,
                    "filename": job["filename"],
                    "tillicum_path": str(PDF_DIR / f"{jid}.pdf"),
                    "path_parts": job["path_parts"],
                    "pages": [{"page": 0, "lines": text_lines}],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
            out_f.flush()
            os.fsync(out_f.fileno())

            n = (bi + 1) * args.batch
            rate = done / (time.time() - t_start)
            eta_h = (len(jobs) - n) / rate / 3600 if rate else float("inf")
            print(f"{tag} batch {bi+1}/{len(batches)} | done={done} fail={failed} "
                  f"| {rate:.2f} pdf/s | eta {eta_h:.1f}h", flush=True)

    elapsed = (time.time() - t_start) / 3600
    print(f"{tag} FINISHED in {elapsed:.2f}h | done={done} failed={failed}", flush=True)

if __name__ == "__main__":
    main()
