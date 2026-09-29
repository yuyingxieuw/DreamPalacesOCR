# OCR Pipeline — Benchmarks

Performance notes for the Dream Palace OCR pipeline (Surya, run on an HPC GPU
cluster via Slurm). Optimizations are grouped into **single-GPU efficiency**
(getting more out of each GPU) and **horizontal scaling** (running across
multiple GPUs).

> **A note on metrics.** Early runs reported throughput as `pdf/s`, which turned
> out to be unreliable: pages per PDF vary widely (in one run, three batches of
> 8 PDFs each contained 2,298 / 4,119 / 3,961 lines — nearly 2× spread). All
> throughput figures below therefore use **lines/second (it/s)** from Surya's
> recognition step, a workload-normalized unit. Multi-GPU results use wall-clock
> time over a fixed file count.

## Summary

| Optimization                       | Type                  | Result                                        |
| ---------------------------------- | --------------------- | --------------------------------------------- |
| pdf2image → pypdfium2              | Single-GPU (I/O)      | **2.18× faster PDF rendering**                |
| DPI 300 → 200 (accuracy validated) | Single-GPU (compute)  | recognition throughput **~82 → ~143 lines/s** |
| Batching (1 → 8)                   | Single-GPU (GPU util) | throughput stabilized, denser GPU usage       |
| Prefetching                        | Single-GPU (GPU util) | reduced GPU idle time between batches         |
| 8-way GPU sharding                 | Horizontal scaling    | 800 files in ~6.8 min, balanced across 8 GPUs |

---

## Single-GPU efficiency

### PDF rendering: pdf2image → pypdfium2

Measured on a fixed set of 200 PDFs:

| Engine    | Total time | Rate         |
| --------- | ---------- | ------------ |
| pdf2image | 57.96s     | 3.45 files/s |
| pypdfium2 | 26.56s     | 7.53 files/s |

**≈ 2.18× faster** rendering by switching the PDF-to-image backend, removing the
subprocess/Poppler overhead of pdf2image.

### DPI tuning: 300 → 200

Recognition throughput (Surya `Recognizing Text`, lines/s):

| DPI | Throughput                  |
| --- | --------------------------- |
| 300 | ~77–87 lines/s (≈82 avg)    |
| 200 | ~138–152 lines/s (≈143 avg) |

Rendering at 200 DPI instead of 300 nearly doubles recognition throughput
(**~1.74×**) with **no meaningful loss in OCR accuracy** (validated on sample
pages) — 300 DPI was oversampling for this corpus. The `Detecting bboxes` step
was unaffected; the gain is in text recognition.

### Batching (batch size 1 → 8)

On a 1,000-file sample, increasing batch size raised and stabilized throughput
(64–104 → 75–87 lines/s) by packing more work into each GPU call and reducing
per-inference overhead. The main benefit is denser, more continuous GPU usage.

### Prefetching

Overlapping data loading with GPU inference reduced GPU idle time between
batches (shorter idle gaps observed in `nvidia-smi` utilization sampling),
keeping the GPU fed more continuously.

---

## Horizontal scaling

### 8-way GPU sharding

The manifest is split into N shards, each pinned to its own GPU and run in
parallel.

**Run (job 328329):** 8 shards × 100 files = **800 files** across 8 GPUs.

| Metric           | Value                                          |
| ---------------- | ---------------------------------------------- |
| Total wall-clock | ~6 min 46 s                                    |
| Per shard        | 100 files in ~0.08h each                       |
| Load balance     | all 8 shards finished within the same interval |

Even completion times across shards indicate the manifest was partitioned
evenly, with no straggler GPU holding up the run.

---

## Methodology notes / limitations

- Early measurements used `pdf/s` and were later found unreliable (see metrics
  note above); figures here use lines/s or wall-clock over fixed file counts.
- GPU-utilization observations (batching, prefetch) are from `nvidia-smi`
  sampling and are qualitative, not precise utilization measurements.
- Single-GPU optimizations were benchmarked per-stage rather than as one
  controlled end-to-end v1-vs-v3 run. <!-- TODO: confirm -->
- DPI 300 and DPI 200 throughput come from different file samples; comparison is
  valid because the unit (lines/s) is workload-normalized. <!-- TODO: confirm if same set -->

## Roadmap

- Re-benchmark end-to-end (v1 vs current) on one fixed set with page-normalized
  throughput, to report a single clean overall speedup.
