# Qwen3-0.6B — Results

Drop the real outputs from `notebooks/task2_qwen_training.ipynb` here, once per backend:

```
qwen_0.6b_cpu_log.txt        ← raw stdout from the CPU run
qwen_0.6b_cpu_metrics.csv    ← parsed per-step metrics (CPU)
qwen_0.6b_gpu_log.txt
qwen_0.6b_gpu_metrics.csv
qwen_0.6b_tpu_log.txt
qwen_0.6b_tpu_metrics.csv
summary_cpu.json             ← contains both qwen_0.6b + qwen_1b summary for that backend
summary_gpu.json
summary_tpu.json
screenshots/                 ← optional: Colab screenshots (runtime type, nvidia-smi, final step output)
```

These come directly out of the notebook's "Download results" cell (unzip the downloaded archive here).
