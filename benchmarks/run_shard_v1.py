## file for OCR pdf at Tillicum with multiple GPU 
## file format in manifest ::
## {"file_id": "1Hr9Zo-LXK4xXEttWKIMl_1Zqxt1Pvwj1", "filename": "10_29_1945_Page-399.pdf", "path_parts": ["Mid-West", "Ohio", "The daily bulletin", "1945"]}

import argparse 
import json
import os 
import sys
import time 
import traceback
from pathlib import Path

def load_shard(manifest: Path, shard_index: int, num_shards:int):
    """
    allocate jobs based on shard_index and number of shards 
    return: json 
    [{"file_id": "1Hr9Zo-LXK4xXEttWKIMl_1Zqxt1Pvwj1", "filename": "10_29_1945_Page-399.pdf", "path_parts": ["Mid-West", "Ohio", "The daily bulletin", "1945"]}]
    """
        
    if num_shards <= 0:
        raise SystemExit("num_shards must be a positive integer")
    
    jobs = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            if i % num_shards == shard_index:
                jobs.append(json.loads(line))
    return jobs


def load_completed_ids(results_path: Path) -> set:
    """Collect ids already present in this shard's results file.

    Only whole lines (ending in newline) that parse as JSON with an "id" are
    trusted. A torn final line from a killed worker is ignored, so that PDF is
    reprocessed — safe, just produces a duplicate line to dedupe on read.

    return: set ("1Hr9Zo-LXK4xXEttWKIMl_1Zqxt1Pvwj1",  "1uj2JzAHJbfVyD_EFPoGmtt3VRnwqPhSw")
    """
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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[shard {args.shard_index}/{args.num_shards}]"

    # confirm the launcher pinned to exactly one GPU
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"{tag} CUDA_VISIBLE_DEVICES={visible}", flush=True)

    if visible == "<unset>":
        raise SystemExit(f"{tag} ERROR: CUDA_VISIBLE_DEVICES not set — "
                        f"launcher failed to pin a GPU")
    if "," in visible:
        raise SystemExit(f"{tag} ERROR: this worker sees multiple GPUs "
                        f"({visible}); expected exactly one")

    # One results file per shard; a writeen line doubles as the done-marker
    results_path = args.out_dir / f"shard_{args.shard_index}.results.jsonl"
    fail_log = args.out_dir / f"shard_{args.shard_index}.failed.log"

    completed = load_completed_ids(results_path)
    print(f"{tag} {len(completed)} jobs already completed (from results file)",
            flush=True)

    # ---- Import Surya only AFTER the GPU is pinned ----
    # Load the models once, reuse across every PDF in the shard. Re-loading per
    # file would dominate runtime.
    import surya
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from pdf2image import convert_from_path

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    jobs = load_shard(args.manifest, args.shard_index, args.num_shards)
    print(f"{tag} {len(jobs)} jobs assigned", flush=True)

    done = skipped = failed = 0
    t_start = time.time()

    with results_path.open("a") as out_f, fail_log.open("a") as fail_f:
        for n, job in enumerate(jobs, 1):
            jid = job["file_id"]
            org_path_parts = job["path_parts"]
            file_name = job["filename"]

            # in file pdfs content path are 
            # /gpfs/scrubbed/yxie993/pdfs/1zzw6lwl8XgKpjQJoIOF3oEMhAbEQ7Ulm.pdf
            path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{jid}.pdf"
            
            if jid in completed:
                skipped +=1
                continue
            
            try:
                images = convert_from_path(str(path), dpi = 300)
                pages = []
                for page_no, img in enumerate(images):
                    preds = recognition([img], det_predictor=detection)
                    text_lines = [
                        line.text
                        for pred in preds
                        for line in pred.text_lines
                    ]
                    pages.append({"page": page_no, "lines": text_lines})
                
                record = {"id": jid, "filename":file_name, "tillicum_path": str(path), "path_parts": org_path_parts, "pages": pages}

                # Write the whole record as ONE line, then force to disk. Writing
                # a single line keeps the append atomic-ish and easy to parse.

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())

                completed.add(jid)
                done += 1
            
            except Exception:
                err = traceback.format_exc().strip().replace("\n", " | ")
                fail_f.write(f"{jid}\t{path}\t{err}\n")
                fail_f.flush()
                failed += 1

            if n % 25 == 0:
                rate = n / (time.time() - t_start)
                eta_h = (len(jobs) - n) / rate / 3600 if rate else float ("inf")
                print (
                    f"{tag} {n}/{len(jobs)} | done={done} skip={skipped} "
                    f"fail={failed} | {rate:.2f} pdf/s | eta {eta_h:.1f}h",
                    flush=True,
                )

    elapsed = (time.time() - t_start) / 3600
    print(
        f"{tag} Finished in {elapsed:.2f}h | done = {done}",
        f"skipped={skipped} failed={failed}",
        flush=True
    )

if __name__ == "__main__":
    main()
