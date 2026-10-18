#!/usr/bin/env python3
"""
--- results.jsonl will be stored in R2, so need to change this db to a index db-- 
Migrate OCR results (results.jsonl) from Tillicum into Supabase Postgres.

Each line of results.jsonl looks like:
    {
      "id": "<jid>",
      "filename": "10_29_1945_Page-399.pdf",
      "tillicum_path": "/gpfs/.../10_29_1945_Page-399.pdf",
      "path_parts": ["Mid-West", "Ohio", "The daily bulletin", "1945"],
      "pages": [{"page": 0, "lines": [...]}, ...]
    }

Design notes:
- Idempotent: file_id is the primary key and we INSERT ... ON CONFLICT DO
  NOTHING, so re-running is safe — already-inserted rows are skipped. Just rerun
  the whole script if it's interrupted.
- Batched: rows are inserted and committed in batches (default 1000) instead of
  one giant 190k-row transaction. Keeps memory bounded and means an interruption
  only loses the last uncommitted batch, which the next run re-inserts.
- Raw data preserved: path_parts and pages are stored as jsonb. region/state/
  newspaper/year are *derived* from path_parts for convenient querying, but the
  original path_parts stays so they can be recomputed if the derivation was off.

Usage if run in tillicum:
    export SUPABASE_DB_URL="postgresql://...key..."
    python to_supabase.py --result /gpfs/scrubbed/yxie993/results.jsonl
    python to_supabase.py --result /gpfs/scrubbed/yxie993/results.jsonl --batch-size 50
"""
import argparse
import json
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

TABLE = "newspaper_archive"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    file_id    text PRIMARY KEY,
    filename   text,
    path_parts jsonb,
    region     text,
    state      text,
    newspaper  text,
    year       text,
    content    jsonb
);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE}
    (file_id, filename, path_parts, region, state, newspaper, year, content)
VALUES %s
ON CONFLICT (file_id) DO NOTHING
"""

# Template describes ONE row's placeholders for execute_values. jsonb columns
# get %s too — psycopg2 sends the json string and Postgres casts it.
ROW_TEMPLATE = "(%s, %s, %s, %s, %s, %s, %s, %s)"


def transform(r: dict) -> tuple:
    """Turn one JSONL record into a row tuple matching INSERT_SQL's columns."""
    path_parts = r.get("path_parts") or []
    region    = path_parts[0] if len(path_parts) > 0 else None
    state     = path_parts[1] if len(path_parts) > 1 else None
    newspaper = path_parts[2] if len(path_parts) > 2 else None
    year      = path_parts[3] if len(path_parts) > 3 else None

    return (
        r.get("id"),
        r.get("filename"),
        json.dumps(path_parts, ensure_ascii=False),   # -> jsonb
        region,
        state,
        newspaper,
        year,
        json.dumps(r.get("pages"), ensure_ascii=False),  # -> jsonb
    )


def flush(cur, conn, batch):
    """Insert one batch of row tuples and commit."""
    if not batch:
        return 0
    execute_values(cur, INSERT_SQL, batch, template=ROW_TEMPLATE)
    conn.commit()
    return len(batch)


def main():
    ## if run on tillicum uncomment the following
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--result", required=True, type=Path,
    #                 help="path to merged results.jsonl")
    # ap.add_argument("--batch-size", type=int, default=1000)
    # args = ap.parse_args()

    # if not args.result.exists():
    #     raise SystemExit(f"Result file not found: {args.result}")
    INPUT_PATH = "/Users/xy/Documents/workspace/DreamPalacesOCR/data/ocr_merged.jsonl"
    BATCH_SIZE= 500

    load_dotenv()
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit("SUPABASE_DB_URL not set (check your .env)")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Make sure the table exists before inserting.
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    read = inserted_batches = bad = 0
    batch = []

    total = 197364
    with open(INPUT_PATH,"r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            batch.append(transform(obj))
            read += 1

            if len(batch) >= BATCH_SIZE:
                inserted_batches += flush(cur, conn, batch)
                batch = []
                if read % 5000 < BATCH_SIZE:
                    pct = read / total * 100 if total else 0
                    print(f"  {read}/{total} ({pct:.1f}%) read, {inserted_batches} committed",
                          flush=True)

    # Final partial batch.
    inserted_batches += flush(cur, conn, batch)

    cur.close()
    conn.close()

    print(f"\nDone.")
    print(f"  records read from {INPUT_PATH}: {read}")
    print(f"  rows sent to Postgres:                {inserted_batches}")
    print(f"  skipped (bad JSON lines):             {bad}")
    print(f"\nNote: rows already present were skipped via ON CONFLICT DO NOTHING,")
    print(f"so the actual inserted count may be lower. Verify with:")
    print(f'  SELECT count(*) FROM {TABLE};')


if __name__ == "__main__":
    main()
