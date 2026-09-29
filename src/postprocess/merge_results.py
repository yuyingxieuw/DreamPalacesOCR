#!/usr/bin/env python3
"""
Merge the per-shard results into one deduplicated JSONL.
 
Run this once after all shards finish. It reads shard_*.results.jsonl and writes
a single results.jsonl where each id appears exactly once. If a PDF was
processed more than once (e.g. reprocessed after a kill mid-write), the LAST
occurrence wins — that's the newest, complete result.
 
    python merge_results.py --out-dir ocr_out --out results.jsonl
 
Memory note: this holds one line per unique id in memory. At ~200k PDFs that's
fine. If a single result line is huge (very long OCR text) and memory is tight,
switch to the two-pass variant noted at the bottom.
"""
import argparse
import json
from pathlib import Path
 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required= True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    result_files = sorted(args.out_dir.glob("shard_*.results.jsonl"))
    if not result_files:
        raise SystemExit(f"No shard_*.results.jsonl found in {args.out_dir}")

    # id -> raw line. Later occurrences overwrite earlier ones, so the last
    # write for any id wins — exactly the dedupe rule we want.

    latest = {}
    total_lines = 0
    for rf in result_files:
        with rf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jid = obj.get("id")
                if not jid:
                    continue
                latest[jid] = line
                total_lines += 1

    with args.out.open("w") as out_f:
        for line in latest.values():
            out_f.write(line + "\n")

    dupes = total_lines - len(latest)
    print(f"read {total_lines} lines from {len(result_files)} shards")
    print(f"wrote {len(latest)} unique records to {args.out}")
    print(f"deduplicated {dupes} duplicate line(s)")


if __name__ == "__main__":
    main()
