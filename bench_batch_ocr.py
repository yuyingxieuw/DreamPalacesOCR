"""
Bench time batch job to recgoniction in surya
Run at tillicum conda surya

How to use: 
Use two terminal
Terminal ONE:
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 

python bench_batch_ocr.py --manifest manifest.jsonl --out-dir /tmp/test_b1  --shard-index 0 --num-shards 1000 --batch 1
python bench_batch_ocr.py --manifest manifest.jsonl --out-dir /tmp/test_b8  --shard-index 0 --num-shards 1000 --batch 8
python bench_batch_ocr.py --manifest manifest.jsonl --out-dir /tmp/test_b16 --shard-index 0 --num-shards 1000 --batch 16

Terminal TWO:
nvidia-smi dmon -s u #print GPU usage rate per second

Results:
...
"""


import argparse, json, os, sys, time, traceback
from pathlib import Path

def load_shard(manifest, shard_index, num_shards):
    if num_shards <= 0:
        raise SystemExit("num_shards must be a positive integer")
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
    ap.add_argument("--batch", type=int, default=8, help="page number/batch, test 8/16/32")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[shard {args.shard_index}/{args.num_shards}]"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"{tag} CUDA_VISIBLE_DEVICES={visible} batch={args.batch}", flush=True)
    if visible == "<unset>":
        raise SystemExit(f"{tag} ERROR: CUDA_VISIBLE_DEVICES not set")
    if "," in visible:
        raise SystemExit(f"{tag} ERROR: sees multiple GPUs ({visible})")

    results_path = args.out_dir / f"shard_{args.shard_index}.results.jsonl"
    fail_log = args.out_dir / f"shard_{args.shard_index}.failed.log"
    completed = load_completed_ids(results_path)
    print(f"{tag} {len(completed)} already completed", flush=True)

    # Import surya AFTER GPU pinned
    from surya.input.processing import open_pdf, get_page_images
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    jobs = load_shard(args.manifest, args.shard_index, args.num_shards)
    jobs = [j for j in jobs if j["file_id"] not in completed]
    print(f"{tag} {len(jobs)} jobs to do (after skip)", flush=True)

    done = failed = 0
    t_start = time.time()

    def render_job(job):
        """render fist page of each PDF; return None if failed"""
        path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{job['file_id']}.pdf"
        doc = open_pdf(str(path))
        img = get_page_images(doc, [0], dpi=args.dpi)[0]
        doc.close()
        return img

    with results_path.open("a") as out_f, fail_log.open("a") as fail_f:
        # divide work based on batch size 
        for start in range(0, len(jobs), args.batch):
            batch_jobs = jobs[start:start + args.batch]

            # 1) Render current patch (sequentially; this step only tests batch recognition performance)  
            imgs, ok_jobs = [], []
            for job in batch_jobs:
                path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{job['file_id']}.pdf"
                try:
                    imgs.append(render_job(job))
                    ok_jobs.append(job)
                except Exception:
                    err = traceback.format_exc().strip().replace("\n", " | ")
                    fail_f.write(f"{job['file_id']}\t{path}\t{err}\n")
                    fail_f.flush()
                    failed += 1

            if not imgs:
                continue

            # 2) send batch imgs to recognition 
            try:
                preds = recognition(imgs, det_predictor=detection)
            except Exception:
                
                err = traceback.format_exc().strip().replace("\n", " | ")
                for job in ok_jobs:
                    path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{job['file_id']}.pdf"
                    fail_f.write(f"{job['file_id']}\t{path}\tBATCH_FAIL {err}\n")
                    failed += 1
                fail_f.flush()
                continue

            # 3) write result back to record

            for job, pred in zip(ok_jobs, preds):
                jid = job["file_id"]
                path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{jid}.pdf"
                text_lines = [line.text for line in pred.text_lines]
                record = {
                    "id": jid,
                    "filename": job["filename"],
                    "tillicum_path": str(path),
                    "path_parts": job["path_parts"],
                    "pages": [{"page": 0, "lines": text_lines}],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1

            out_f.flush()
            os.fsync(out_f.fileno())

            n = start + len(batch_jobs)
            rate = n / (time.time() - t_start)
            eta_h = (len(jobs) - n) / rate / 3600 if rate else float("inf")
            print(f"{tag} {n}/{len(jobs)} | done={done} fail={failed} "
                  f"| {rate:.2f} pdf/s | eta {eta_h:.1f}h", flush=True)
            
    elapsed = (time.time() - t_start) / 3600
    print(f"{tag} Finished in {elapsed:.2f}h | done={done} failed={failed}", flush=True)

if __name__ == "__main__":
    main()
