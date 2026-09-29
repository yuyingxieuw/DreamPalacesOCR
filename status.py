#!/usr/bin/env python3
"""
Report progress, surface failures, and check for duplicate ids.

    python status.py --manifest manifest.jsonl --out-dir ocr_out
    python status.py --manifest manifest.jsonl --out-dir ocr_out --show-failures
    python status.py --manifest manifest.jsonl --out-dir ocr_out --check-dupes

Progress is counted from the per-shard results files (shard_<k>.results.jsonl):
a few file reads, not a scan of hundreds of thousands of small files.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def iter_ids(results_path: Path):
    if not results_path.exists():
        return
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = obj.get("id")
            if jid:
                yield jid


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--check-dupes", action="store_true")
    args = ap.parse_args()

    total = sum(1 for _ in args.manifest.open())
    result_files = sorted(args.out_dir.glob("shard_*.results.jsonl"))
    fail_logs = sorted(args.out_dir.glob("shard_*.failed.log"))

    # Unique completed ids across all shards.
    all_ids = Counter()
    for rf in result_files:
        for jid in iter_ids(rf):
            all_ids[jid] += 1

    done_unique = len(all_ids)
    failed = sum(count_lines(p) for p in fail_logs)
    dupes = {jid: c for jid, c in all_ids.items() if c > 1}

    pct = 100 * done_unique / total if total else 0
    print(f"total:          {total}")
    print(f"done (unique):  {done_unique} ({pct:.1f}%)")
    print(f"failed:         {failed}")
    print(f"remaining:      {total - done_unique - failed}")
    if dupes:
        print(f"DUPLICATE ids:  {len(dupes)}  (processed more than once)")

    print("\nper-shard result lines:")
    for rf in result_files:
        print(f"  {rf.name}: {count_lines(rf)}")

    if args.check_dupes:
        if not dupes:
            print("\nNo duplicate ids — every PDF appears exactly once.")
        else:
            print(f"\n{len(dupes)} duplicate ids (id: times seen), first 20:")
            for jid, c in list(dupes.items())[:20]:
                print(f"  {jid}: {c}")
            print("\nDuplicates are harmless: merge_results.py keeps the last "
                  "occurrence of each id.")

    if args.show_failures and fail_logs:
        print("\n--- sample failures (first 20 across shards) ---")
        shown = 0
        for p in fail_logs:
            with p.open() as f:
                for line in f:
                    if line.strip():
                        print(line.rstrip())
                        shown += 1
                        if shown >= 20:
                            break
            if shown >= 20:
                break
        print("\nFailed ids are absent from the results files, so they are")
        print("reprocessed automatically on resubmit. Clear the failed logs")
        print("first to keep the list clean:")
        print(f"  rm {args.out_dir}/shard_*.failed.log && sbatch submit.slurm")


if __name__ == "__main__":
    main()
