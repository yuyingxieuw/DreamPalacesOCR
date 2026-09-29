# Dream Palace — OCR Pipeline

Large-scale OCR pipeline that digitizes a **197,364-page** historical Black
newspaper archive into searchable, structured text. Built on
[Surya](https://github.com/VikParuchuri/surya), run on an HPC GPU cluster via
Slurm, and engineered for high throughput on degraded, multi-column scans.

Part of the [Dream Palace](../..) project. <!-- TODO: link to DreamPalaceOverview repo -->

---

## What it does

The pipeline takes source PDFs and produces structured OCR output loaded into a
Postgres database, in four stages:
| Stage | What happens | Code |
|-------|--------------|------|
| **Ingest** | Pull source PDFs from Google Drive to cluster storage | `src/ingest/` |
| **OCR** | Render PDF pages and run Surya text detection + recognition, sharded across GPUs | `src/ocr/` |
| **Postprocess** | Merge and structure per-shard OCR output | `src/postprocess/` |
| **Load** | Write structured results into Supabase (Postgres) | `src/load/` |

---

## Performance

The pipeline was re-engineered from a naive baseline for substantially higher
throughput. Optimizations fall into two categories:

**Single-GPU efficiency** — getting more work out of each GPU:

- **PDF rendering:** replaced `pdf2image` with `pypdfium2` → **2.18× faster rendering**
- **DPI tuning:** 300 → 200 DPI (validated: no meaningful accuracy loss) → recognition throughput **~82 → ~143 lines/s**
- **Batching & prefetching:** larger batches and overlapping data loading with
  inference keep the GPU continuously fed, reducing idle time between batches

**Horizontal scaling** — running across the cluster:

- **Multi-GPU sharding:** the manifest is split into N shards, each pinned to its
  own GPU and run in parallel (e.g. 800 files across 8 GPUs in ~6.8 min, evenly
  balanced)

Underlying idea: the per-document flow is **CPU (render) → GPU (OCR) → CPU
(write)**, which leaves both resources idle half the time if run naively.
Prefetching overlaps CPU data preparation with GPU inference, and a faster
render backend keeps the CPU side from bottlenecking the GPU.

See [`benchmarks/`](benchmarks/) for detailed measurements and methodology.

> **Note on metrics:** throughput is reported in **lines/second**, not `pdf/s` —
> pages per PDF vary too widely for `pdf/s` to be a reliable unit. See the
> benchmarks README for details.

---

## Repository structure

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:**

- Python [3.x] <!-- TODO: your version -->
- NVIDIA GPU(s) with CUDA <!-- TODO: confirm / version -->
- A Supabase (Postgres) instance for output
- Access to an HPC cluster with Slurm (for multi-GPU runs)

**Configuration:** [describe how credentials/paths are set — env vars? a config
file? Supabase connection string?] <!-- TODO -->

---

## Usage

A manifest (`manifest.jsonl`) lists the PDFs to process. <!-- TODO: confirm format / how it's generated -->

**Ingest PDFs:**

```bash
python src/ingest/download.py [args]   # TODO: real args
```

**Run OCR (single shard, local):**

```bash
bash scripts/run_local.sh              # TODO: confirm
```

**Run OCR (multi-GPU, on the cluster):**

```bash
sbatch scripts/submit.slurm            # TODO: confirm
```

**Load results into Supabase:**

```bash
python src/load/to_supabase.py [args]  # TODO: real args
```

**Check progress:**

```bash
python scripts/status.py [args]        # TODO: real args
```

---

## Tech

Python · [Surya](https://github.com/VikParuchuri/surya) · pypdfium2 · Slurm ·
Supabase (Postgres)

---

## Roadmap

- Controlled end-to-end re-benchmark (single-GPU, fixed file set, page-normalized
  throughput) for exact per-optimization speedups
- `batch × shard` interaction study (does per-GPU batching hold under multi-GPU
  parallelism, or does CPU/IO become the bottleneck at scale?)
- Unit tests for postprocessing / structuring logic
