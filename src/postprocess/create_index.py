"""
post-process merged_data
this script:
1. load cinema name
2. search each cinema in merged data
3. create dic store the search result:
 [
 {cinema_id: "",
 cinema_name: "",
 search_result: ["pdf_id_1", "pdf_id_2", ...]}
 ]

Matching rules:
- Both the cinema name and the OCR text are normalized the same way
  (lowercase, accents removed, any non-alphanumeric char -> space), so
  "Booker-T", "Brown's" and "Opéra" match their OCR spellings.
- A name only matches whole words: "Rex" matches "REX THEATRE" but not "Rexall".
- One pass over ocr_merged.jsonl checks every cinema at once. Scanning the
  4GB file once per cinema (1400+ times) would take far too long.
- A hit is recorded per PDF (record id), not per page.

Usage (from anywhere):
    python src/postprocess/create_index.py
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from time import time

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "ocr_merged.jsonl"
OUTPUT_PATH = ROOT / "data" / "cinema_index.json"
ENV_PATH = ROOT / "src" / "load" / ".env"


# connect to supabase and get a list of cinemas
def fetch_all_records():
    '''
    Fetch a list of cinema id and name from supabase
    Return: list of dicts like {"id": ..., "name": ...}
    '''
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name FROM dream_palaces order by id")
        try:
            records = cur.fetchall()
        except Exception as e:
            print("Supabase fetch all records failed:", e)
            return []
    finally:
        conn.close()
    return records


def normalize(text):
    """
    lowercase, strip accents (é -> e), turn every non-alphanumeric char into a
    space and collapse runs of spaces. Applied to both names and OCR text.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def clean_name(records):
    """
    take original records
    get each item, remove "cinema", remove whatever is in () or [], split
    alternatives written with | or / ("Queen / New Queen" -> both)
    return: list of {"id", "name", "search_terms": [normalized phrases]}
    """
    cleaned = []
    for r in records:
        name = r["name"] or ""
        name = re.sub(r"\(.*?\)|\[.*?\]", " ", name)
        terms = []
        for part in re.split(r"[|/]", name):
            term = normalize(part)
            # term = re.sub(r"\bcinema\b", " ", term)
            term = " ".join(term.split())
            if term and term not in terms:
                terms.append(term)
        if not terms:
            print(f"  skipped (empty after cleaning): {r['name']!r}")
            continue
        cleaned.append({"id": r["id"], "name": r["name"], "search_terms": terms})
    return cleaned


def build_index(cleaned, input_path=INPUT_PATH):
    """
    take: output of clean_name
    process: read input_path once, check every cinema's search terms per PDF
    return: list of {"cinema_id", "cinema_name", "search_result": [pdf ids]}
    """
    # first word -> [(full phrase, cinema id)]. Looking up a page's word set
    # against these keys is a fast set intersection, and only the few phrases
    # whose first word appears need a substring check.
    by_first_word = defaultdict(list)
    for c in cleaned:
        for term in c["search_terms"]:
            by_first_word[term.split()[0]].append((term, c["id"]))
    first_words = set(by_first_word)

    hits = defaultdict(list)  # cinema id -> [pdf id]
    read = bad = 0
    start = time()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            text = " ".join(
                l for page in obj.get("pages") or [] for l in page.get("lines") or []
            )
            text = f" {normalize(text)} "

            matched = set()
            for w in first_words.intersection(text.split()):
                for term, cinema_id in by_first_word[w]:
                    if cinema_id in matched:
                        continue
                    if term == w or f" {term} " in text:
                        matched.add(cinema_id)
            for cinema_id in matched:
                hits[cinema_id].append(obj["id"])

            read += 1
            if read % 10000 == 0:
                print(f"  searched {read} PDFs ({time() - start:.0f}s)", flush=True)

    print(f"searched {read} PDFs in {time() - start:.0f}s, skipped {bad} bad JSON lines")

    return [
        {
            "cinema_id": c["id"],
            "cinema_name": c["name"],
            "search_result": hits.get(c["id"], []),
        }
        for c in cleaned
    ]


def main():
    load_dotenv(ENV_PATH)
    if not os.environ.get("SUPABASE_DB_URL"):
        raise SystemExit(f"SUPABASE_DB_URL not set (check {ENV_PATH})")

    records = fetch_all_records()
    print(f"fetched {len(records)} cinemas")
    cleaned = clean_name(records)

    index = build_index(cleaned)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    found = sum(1 for i in index if i["search_result"])
    print(f"\n{found}/{len(index)} cinemas found in at least one PDF")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
