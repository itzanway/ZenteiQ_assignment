# ZenteiQ AiTech Innovations — ML Engineer Assignment
## MaxText LLM Training: Dense (Qwen) & MoE (DeepSeek) on CPU / GPU / TPU

**Candidate Submission**

---

## Repository Layout

```
zenteiQ-maxtext-assignment/
├── README.md                        ← This file (overview + quick-start)
├── docs/
│   └── task1_data_formats.md        ← Task 1: MaxText data pipeline analysis
├── configs/
│   ├── qwen/
│   │   ├── qwen3_0.6b_train.yml     ← Qwen3 0.6B base config
│   │   └── qwen3_1b_train.yml       ← Qwen3 ~1B scaled-up config
│   └── deepseek/
│       └── deepseek_moe_sub1b.yml   ← DeepSeek MoE scaled-down (<1B) config
├── notebooks/
│   ├── task2_qwen_training.ipynb    ← Colab notebook: Qwen runs (CPU+GPU+TPU)
│   └── task3_deepseek_training.ipynb← Colab notebook: DeepSeek MoE runs
├── results/
│   ├── qwen_0_6b/                   ← Raw training logs — Qwen 0.6B
│   ├── qwen_1b/                     ← Raw training logs — Qwen ~1B
│   └── deepseek_moe/                ← Raw training logs — DeepSeek MoE <1B
└── analysis/
    └── final_summary.md             ← Task 2 + 3 tables, interpretation, findings
```

---

## Quick-start (Colab)

> All runs were performed in **Google Colab** using free-tier CPU, T4 GPU, and TPU v2-8 runtimes.

### 1. Install MaxText

```bash
# In any Colab cell — works for all three runtimes
!git clone https://github.com/AI-Hypercomputer/maxtext.git
%cd maxtext
!pip install -e ".[tpu]" --quiet          # TPU runtime
# or
!pip install -e ".[gpu]" --quiet          # GPU runtime
# or
!pip install -e "." --quiet               # CPU runtime
```

### 2. Run Qwen 0.6B (example — CPU)

```bash
python -m maxtext.train src/maxtext/configs/base.yml \
  include=src/maxtext/configs/models/qwen3-0.6b.yml \
  steps=50 \
  dataset_type=synthetic \
  base_output_directory=/tmp/maxtext_output \
  run_name=qwen_0.6b_cpu \
  per_device_batch_size=1 \
  max_target_length=512
```

### 3. Run Qwen ~1B (scaled-up)

```bash
python -m maxtext.train src/maxtext/configs/base.yml \
  include=configs/qwen/qwen3_1b_train.yml \
  steps=50 \
  dataset_type=synthetic \
  base_output_directory=/tmp/maxtext_output \
  run_name=qwen_1b_cpu \
  per_device_batch_size=1 \
  max_target_length=512
```

### 4. Run DeepSeek MoE <1B

```bash
python -m maxtext.train src/maxtext/configs/base.yml \
  include=configs/deepseek/deepseek_moe_sub1b.yml \
  steps=50 \
  dataset_type=synthetic \
  base_output_directory=/tmp/maxtext_output \
  run_name=deepseek_moe_sub1b_cpu \
  per_device_batch_size=1 \
  max_target_length=512
```

---

## Status

This repo currently contains the **configs, runnable Colab notebooks, and Task 1 writeup**.
The result tables in `analysis/final_summary.md` and `results/*/` are **fill-in templates** —
they will be populated with real numbers once the notebooks are executed on actual Colab
CPU / GPU / TPU runtimes. See `analysis/final_summary.md` for the table structure and
`results/*/README.md` for what files belong where.

---
