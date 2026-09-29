"""
Bench time pdf2image VS pypdfium2
Run at tillicum conda surya

How to use: 
python bench_render.py --manifest manifest.jsonl --engine pdf2image  --n 200
python bench_render.py --manifest manifest.jsonl --engine pypdfium2  --n 200

Results:
Done engine=pdf2image ok=200 fail=0 total=57.96s per_file=289.8ms rate=3.45 files/s
Done engine=pypdfium2 ok=200 fail=0 total=26.56s per_file=132.8ms rate=7.53 files/s
"""
import argparse, json, time
from pathlib import Path

def load_shard(manifest, shard_index, num_shards):
    jobs = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            if i % num_shards == shard_index:
                jobs.append(json.loads(line))
    return jobs


def render_pdf2image(path, dpi=300):
    from pdf2image import convert_from_path
    return convert_from_path(str(path), dpi=dpi)

def render_pypdfium2(path, dpi=300):
    from surya.input.processing import open_pdf, get_page_images
    doc = open_pdf(str(path))
    images = get_page_images(doc, [0], dpi=300)
    doc.close()
                    
    return images



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--engine", required=True, choices=["pdf2image", "pypdfium2"])
    ap.add_argument("--n", type=int, default=200)   
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    render = render_pdf2image if args.engine == "pdf2image" else render_pypdfium2

    jobs = load_shard(args.manifest, args.shard_index, args.num_shards)[:args.n]
    print(f"engine={args.engine} files={len(jobs)} dpi={args.dpi}", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, job in enumerate(jobs,i):
        path = Path("/gpfs/scrubbed/yxie993/pdfs") / f"{job['file_id']}.pdf"
        try:
            imgs = render(path, dpi=args.dpi)
            _ = imgs[0].size  
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  FAIL {path}: {type(e).__name__}: {e}", flush=True)
        if i % 25 ==0:
            dt = time.time() - t0
            print(f"  {i}/{len(jobs)} ok={ok} fail={fail} {ok/dt:.2f} files/s", flush=True)

    dt = time.time() - t0
    print(f"Done engine={args.engine} ok={ok} fail={fail} "
          f"total={dt:.2f}s per_file={dt/max(ok,1)*1000:.1f}ms "
          f"rate={ok/dt:.2f} files/s", flush=True)

if __name__ == "__main__":
    main()
