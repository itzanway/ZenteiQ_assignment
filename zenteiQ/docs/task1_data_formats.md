# Task 1 — MaxText Data Input Formats

## Overview

MaxText supports three distinct data input pipelines, each designed around a different dataset format philosophy. Before running any training, you need to understand these so you can set `dataset_type` correctly. For Tasks 2 and 3 in this assignment we use `dataset_type: synthetic` — but understanding the other options is critical for real-world training.

---

## The Three Pipelines

### 1. Grain Pipeline (`dataset_type: grain`)
**Recommended for production**

Grain is Google's data-loading library built specifically for scalable ML training. It supports:

- **ArrayRecord** (random-access): Files can be indexed directly, like a dictionary. Any host in a multi-host cluster can jump to any record by its integer index without scanning through the file sequentially. This is the gold standard — it enables fully deterministic training, global shuffling, and graceful handling of uneven data completion across hosts via padding batches.
- **TFRecord** (sequential-access): Data must be read start-to-end. Grain uses file-based sharding to split the dataset across hosts, but this requires the file count to be a clean multiple of the host count, otherwise data gets skewed.
- **Parquet** (sequential-access): Apache columnar format, performant and widely used in data engineering. Also sequential in Grain, so same file-sharding constraint applies.

**Key advantage of Grain+ArrayRecord**: fully deterministic, preemption-resilient. If a training job crashes and restarts, it can resume from exactly the same data position — important for reproducibility.

### 2. Hugging Face Pipeline (`dataset_type: hf`)

Connects directly to the Hugging Face Hub or to local/GCS-stored files in json, parquet, arrow, csv, or txt format. This is convenient for prototyping since you don't need to pre-process or upload data.

- No data download required when using HF Hub
- Supports many formats
- **Limitation**: non-deterministic if a job is preempted (the data cursor is lost). At scale (many hosts), HF Hub access can become a bottleneck because all hosts hit HF's servers. For GCS-based HF datasets this bottleneck goes away.

### 3. TFDS Pipeline (`dataset_type: tfds`)

TensorFlow Datasets pipeline — a more mature, older approach. Only supports TFRecord format. Fast and performant for workloads where preemption is rare, but like HF it is non-deterministic on restart.

### 4. Synthetic (`dataset_type: synthetic`)

No real dataset at all. MaxText generates random integer tokens on the fly. This is purely for benchmarking: it removes the data pipeline from the equation entirely, letting you isolate hardware compute performance from I/O. **This is what we use in Tasks 2 and 3.**

---

## Comparison Table

| Pipeline | `dataset_type` value | Formats | Random Access | Deterministic | Multi-host Scalability | Best for |
|---|---|---|---|---|---|---|
| Grain | `grain` | ArrayRecord, TFRecord, Parquet | ✅ (ArrayRecord) | ✅ (ArrayRecord) | ✅ | Production training |
| Hugging Face | `hf` | JSON, Parquet, Arrow, CSV, TXT, HF Hub | ❌ | ⚠️ (only without preemption) | ⚠️ | Prototyping |
| TFDS | `tfds` | TFRecord | ❌ | ⚠️ (only without preemption) | ✅ | Legacy TFRecord workflows |
| Synthetic | `synthetic` | None (random tokens) | N/A | ✅ | ✅ | Benchmarking hardware, profiling |

---

## Key Insight: Random vs Sequential Access

The most important distinction in MaxText's data formats is whether they support **random access**:

- **Random access (ArrayRecord)**: Think of it like a book with a detailed index. You can instantly open to any page. Multiple readers can read different pages simultaneously without stepping on each other. This solves the "multi-host uniqueness" problem elegantly — Grain assigns each host a disjoint set of indices, and ArrayRecord can serve them concurrently.

- **Sequential access (TFRecord, Parquet, HF formats)**: Think of an audio cassette — you have to fast-forward through to get where you want. Multi-host training must use file-based sharding: each host gets its own subset of files to read linearly. This works, but requires careful file-count planning (`num_files % num_hosts == 0`) and is fragile to preemption.

## Codebase Observations

Exploring the MaxText codebase reveals:

- `src/maxtext/configs/base.yml` sets `dataset_type: c4_array_record` as the default (ArrayRecord via Grain on the C4 dataset), reflecting that Grain is the intended production path.
- The `synthetic` dataset generates tokens using JAX's random number generator with a fixed shape matching your `per_device_batch_size × max_target_length`. Since no I/O happens, the data step is essentially free — this is exactly why it's ideal for isolating compute benchmarks.
- The `generate_padding_batch_train` flag (in `base.yml`) exists specifically for the multi-host case to handle the edge case where one host finishes its data shard before others — it emits zero-weight padding batches to keep all hosts synchronized.
- The `hf` pipeline uses a `hf_path` and `hf_data_dir` parameter pair, while `grain` uses `grain_train_files`. Each pipeline has its own set of accompanying config keys.

## For This Assignment

All training runs in Tasks 2 and 3 use:
```yaml
dataset_type: synthetic
```
This is the correct choice when: (a) you want to benchmark compute throughput, (b) you have no GCS bucket for real data, and (c) you're on Colab without persistent storage. It eliminates data I/O as a confounding variable and lets the hardware metrics speak purely to model compute performance.