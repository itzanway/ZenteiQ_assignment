# Final Summary — MaxText Training Across CPU / GPU / TPU

> **Fill-in template.** Every numeric placeholder below is written as `[FILL: ...]`.
> Replace each with the real value parsed from your `results/*/*_metrics.csv` and
> `summary_*.json` files after running the two notebooks. Do not estimate or
> guess values — pull them directly from your logs.

---

## 1. Qwen3-0.6B (base) — all backends

| Metric | CPU | GPU (T4) | TPU |
|---|---|---|---|
| Total params | 596M (fixed by arch) | 596M | 596M |
| Avg step time (s) | [FILL] | [FILL] | [FILL] |
| Avg TFLOP/s/device | [FILL] | [FILL] | [FILL] |
| Avg tokens/s/device | [FILL] | [FILL] | [FILL] |
| Loss @ step 1 | [FILL] | [FILL] | [FILL] |
| Loss @ step 50 | [FILL] | [FILL] | [FILL] |
| Final grad norm | [FILL] | [FILL] | [FILL] |
| Final param norm | [FILL] | [FILL] | [FILL] |
| Final learning rate | [FILL] | [FILL] | [FILL] |
| Compile/warmup time (s) | [FILL] | [FILL] | [FILL] |
| Total wall time, 50 steps (s) | [FILL] | [FILL] | [FILL] |
| Peak memory (GB) | [FILL] | [FILL] | [FILL] |
| Exit code | [FILL] | [FILL] | [FILL] |

## 2. Qwen ~1B (scaled-up) — all backends

| Metric | CPU | GPU (T4) | TPU |
|---|---|---|---|
| Total params | ~912M (see config derivation) | ~912M | ~912M |
| Avg step time (s) | [FILL] | [FILL] | [FILL] |
| Avg TFLOP/s/device | [FILL] | [FILL] | [FILL] |
| Avg tokens/s/device | [FILL] | [FILL] | [FILL] |
| Loss @ step 1 | [FILL] | [FILL] | [FILL] |
| Loss @ step 50 | [FILL] | [FILL] | [FILL] |
| Final grad norm | [FILL] | [FILL] | [FILL] |
| Final param norm | [FILL] | [FILL] | [FILL] |
| Compile/warmup time (s) | [FILL] | [FILL] | [FILL] |
| Total wall time, 50 steps (s) | [FILL] | [FILL] | [FILL] |
| Peak memory (GB) | [FILL] | [FILL] | [FILL] |
| Exit code | [FILL] | [FILL] | [FILL] |

## 3. DeepSeek MoE <1B — all backends

| Metric | CPU | GPU (T4) | TPU |
|---|---|---|---|
| Total params | ~446M (see config derivation) | ~446M | ~446M |
| Active params/token | ~115M | ~115M | ~115M |
| Avg step time (s) | [FILL] | [FILL] | [FILL] |
| Avg TFLOP/s/device | [FILL] | [FILL] | [FILL] |
| Avg tokens/s/device | [FILL] | [FILL] | [FILL] |
| Loss @ step 1 | [FILL] | [FILL] | [FILL] |
| Loss @ step 50 | [FILL] | [FILL] | [FILL] |
| Load-balance loss (avg) | [FILL] | [FILL] | [FILL] |
| Final grad norm | [FILL] | [FILL] | [FILL] |
| Compile/warmup time (s) | [FILL] | [FILL] | [FILL] |
| Total wall time, 50 steps (s) | [FILL] | [FILL] | [FILL] |
| Peak memory (GB) | [FILL] | [FILL] | [FILL] |
| Exit code | [FILL] | [FILL] | [FILL] |

---

## 4. Interpretation

### 4.1 Why do numbers differ across CPU vs GPU vs TPU?

*(Write this after you have real numbers — a few anchor points to address with your own data:)*

- CPU has no specialized matrix-multiply unit; XLA falls back to vectorized scalar loops, so step time is dominated by raw FLOP throughput at low utilization. Expect CPU step times to be the slowest by 1-2 orders of magnitude.
- GPU (T4) has Tensor Cores for matmul but only ~16GB VRAM and an older architecture (Turing) compared to TPU's purpose-built matmul units; expect GPU step time and TFLOP/s to sit between CPU and TPU.
- TPU (v2-8, single core typically used by default Colab JAX setup) is purpose-built for the kind of dense matmuls transformers require, and JAX/XLA targets it natively — expect lowest step time and highest TFLOP/s/device.
- Compile time differs sharply too: XLA's TPU compiler does more aggressive fusion/sharding analysis, so first-step compile latency on TPU is often the largest of the three, even though steady-state throughput is the best.

*(Replace this with your own observed numbers and explanation once you have them — e.g. "On my run, TPU step time was Nx faster than GPU and Mx faster than CPU, consistent with...")*

### 4.2 Why do they differ between the base (0.6B) and scaled-up (1B) model?

- Parameter count roughly +53% (596M → 912M) increases FLOPs per step roughly proportionally (forward+backward FLOPs scale ~linearly with non-embedding parameter count, for fixed sequence length and batch size).
- Wider `base_emb_dim` (1024→1536) and `base_mlp_dim` (3072→4096) increase matmul dimensions, which can *improve* hardware utilization (larger matmuls amortize fixed overhead better) even though total compute increases — this is why TFLOP/s/device sometimes goes *up* with model size on the same hardware, even as step time also goes up.
- Memory pressure increases with model size; on GPU (16GB) this is the constraint most likely to bite first — watch for OOM or for needing to reduce `per_device_batch_size`.

*(Fill in with your actual loss/step-time deltas once collected.)*

### 4.3 What changed in the config to scale the model up, and why?

See full derivation and reasoning in [`configs/qwen/qwen3_1b_train.yml`](../configs/qwen/qwen3_1b_train.yml). Summary:

| Param | 0.6B | ~1B | Reasoning |
|---|---|---|---|
| `base_emb_dim` | 1024 | 1536 | Widest single lever on parameter count; scales attention + embedding table |
| `base_mlp_dim` | 3072 | 4096 | Maintained ~2.67x emb_dim ratio (FFN is ~2/3 of non-embedding params) |
| `base_num_decoder_layers` | 28 | 24 | Reduced depth to compensate for added width, targeting ~1B total |
| `head_dim` | 64 | 96 | Kept per-head capacity proportional to wider emb_dim (emb_dim / num_heads) |

### 4.4 MoE vs Dense — how do the numbers compare, and why?

*(Fill in after running both. Discussion points to address with your real numbers:)*

- At similar **total** parameter count, MoE models have far fewer **active** parameters per token (here ~115M active vs ~446M total — a ~3.9x sparsity ratio from top-2-of-8 routing). This typically means MoE step time / FLOP cost is closer to a dense model of the *active* size, not the *total* size — compare your DeepSeek MoE step time against Qwen 0.6B (596M dense) rather than against a hypothetical 446M dense model.
- MoE introduces an additional loss term (load-balance loss) and additional all-to-all-style routing/dispatch overhead even on a single device (gather/scatter for expert assignment), which can make per-step latency *higher* than a similarly-sized dense model despite fewer active FLOPs, especially on CPU/GPU where `sparse_matmul=False` (dense matmul with masking) is used here for portability.
- Expect MoE's relative speed advantage (if any) to be most visible on TPU, where XLA can fuse the masked computation efficiently; on CPU the routing overhead may dominate.

---

## 5. Unexpected findings

*(This section is meant to be discovered, not authored in advance. Likely candidates based on common MaxText/Colab friction points — confirm which actually happened to you and remove/replace the rest:)*

- TPU install in Colab may require a specific `jax[tpu]` wheel version pinned to the Colab TPU runtime's libtpu version — mismatches commonly throw cryptic XLA errors on first compile.
- GPU runs may hit `RESOURCE_EXHAUSTED` (OOM) at `per_device_batch_size=1` for the 1B model depending on exact Colab GPU allocated (T4 vs L4 vs A100 vs whatever Colab gives that session) — note whatever you actually got from `nvidia-smi`.
- CPU compile time for the larger model may be disproportionately long relative to its modest size increase, because XLA's CPU backend does less aggressive operation fusion than TPU/GPU backends.
- DeepSeek's MoE block may require additional MaxText-specific config keys not anticipated in `configs/deepseek/deepseek_moe_sub1b.yml` if the installed MaxText version's `deepseek2-lite` model file expects different YAML keys — check the actual error message if `python -m maxtext.train` fails on first attempt and adjust.

---

## 6. Screenshots

*(Insert screenshots here with captions, e.g.:)*

**Figure 1.** Colab Runtime > Change runtime type dialog showing TPU v2-8 selected.

**Figure 2.** `nvidia-smi` output confirming T4 GPU allocation before the GPU run.

**Figure 3.** Final 5 lines of stdout from the Qwen 0.6B TPU run, showing step 50 metrics.
