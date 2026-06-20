# ZenteiQ MaxText Assignment — Samajhne ke liye Reference Doc

> Yeh doc copy-paste ke liye nahi hai. Yeh padhke tujhe samajhna hai ki kya hua aur kyu hua —
> taaki discussion mein tu confidently answer de sake.

---

## Assignment ka ek line summary

MaxText (Google ka JAX-based LLM training framework) use karke 3 models train karne the:
- **Qwen3-0.6B** (dense, baseline)
- **Qwen3 ~1.7B** (same architecture, manually scaled up)
- **DeepSeek MoE sub-1B** (Mixture of Experts, shrunk from 16B)

Teeno ko 3 hardware backends pe run karna tha: **CPU**, **GPU (T4)**, **TPU (v5e-1)**

Dataset: **synthetic** (fake random data, no real text needed — pure benchmarking)

---

## Task 1 — Data Formats (ye poochha jayega)

MaxText mein `dataset_type` config se data pipeline choose hoti hai. 4 options hain:

| Type | Kya hai | Kab use karo |
|---|---|---|
| `synthetic` | Random integer tensors, no file needed | Hardware benchmarking — I/O overhead zero |
| `grain` | Google ka production pipeline, ArrayRecord format | Real training at scale |
| `hf` | HuggingFace Datasets — Parquet, JSON, CSV | Prototyping, quick experiments |
| `tfds` | TensorFlow Datasets — TFRecord only | Legacy pipelines |

**Humne `synthetic` kyun use kiya?**
Kyunki hume model train nahi karna tha — sirf hardware ka throughput measure karna tha.
Synthetic data se disk I/O completely eliminate hoti hai, isliye jo bhi metrics aate hain
woh pure compute performance show karte hain.

**Discussion mein pooch sakte hain:**
- "Grain aur HuggingFace mein kya fark hai?"
  → Grain globally shuffled hai aur preemption-safe hai (deterministic indices).
    HuggingFace sequential read karta hai, preemption pe restart hoga.
- "Production mein kaunsa use karoge?"
  → Grain with ArrayRecord format.

---

## CPU Backend — Kyu Fail Hua

### Problem 1: OOM (Out of Memory) — Exit=-9

```
Total memory size: 11.4 GB
Colab CPU RAM: ~13 GB total
```

**Kya hua:** Qwen3-0.6B (0.596 billion parameters) ko train karne ke liye:
- Model weights: ~2.4 GB (float32)
- Gradients: ~2.4 GB (same size as weights)
- AdamW optimizer state: ~4.8 GB (2x weights — momentum + variance)
- Activation memory: remaining

Total ~11.4 GB. Colab ke paas 13 GB total hai jo OS aur Python process bhi share karti hai.
OS ne process ko SIGKILL (-9) se maar diya.

**Fix jo try kiya:** `max_target_length=32` (sequence length kam karo) + `remat_policy=minimal`
Yeh bhi kaam nahi aaya kyunki notebook ne change nahi kiya config mein.

**Concept samajhna:** AdamW optimizer 2x memory isliye leta hai kyunki woh har parameter ke liye
do extra values store karta hai — first moment (momentum) aur second moment (variance).
Isliye training inference se 3-4x zyada memory leta hai.

### Problem 2: Pallas CPU Kernel — DeepSeek fail

```
ValueError: Only interpret mode is supported on CPU backend.
```

**Kya hua:** MaxText attention operation ke liye `gpu_pallas_attention.mha()` call karta hai
even on CPU. Pallas ek JAX kernel compiler hai jo GPU/TPU ke liye optimized hai.
CPU pe sirf "interpret mode" supported hai — actual kernel compile nahi hota.

**Fix:** `attention=dot_product` pass karo — yeh pure JAX fallback hai jo CPU pe kaam karta hai.

---

## GPU Backend (T4) — Kya Kaam Kiya, Kya Nahi

### ✅ Qwen3-0.6B — SUCCEEDED

```
Parameters: 0.596B
Avg step time: 1.55s
TFLOP/s/device: 2.48
Tokens/s/device: 661.77
Memory used: 10.8 GB (T4 has 15 GB VRAM)
Wall time: 104.4s for 50 steps
```

**Yeh kaam kiya kyunki:** GPU ka VRAM system RAM se alag hota hai.
T4 ka 15 GB VRAM dedicated hai sirf GPU computation ke liye.
0.6B model 10.8 GB mein fit ho gaya.

### ❌ Qwen3-1.7B — OOM

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.16GiB
Input/output arguments: 19.3 GB
T4 VRAM: 15 GB
```

**Kya hua:** 1.7B model ke weights + gradients + optimizer state ~19 GB chahiye.
T4 ke paas sirf 15 GB hai. Simple arithmetic — model fit nahi hota.

**Fix possible tha:** `max_target_length=256` se try kar sakte the (sequence length kam karo).

### ❌ DeepSeek MoE — Triton/MegaBlox Kernel

```
Triton support is only enabled for Ampere GPUs (compute capability 8.0) and up,
but got compute capability 7.5.
```

```
NotImplementedError: dynamic grid bounds not supported in the Triton backend
```

**Kya hua:** Do alag problems:
1. T4 Turing architecture (compute 7.5) hai. Triton ko Ampere (8.0+) chahiye.
2. MoE (Mixture of Experts) ke liye MaxText MegaBlox GMM kernel use karta hai
   jo dynamic grid bounds use karta hai — Triton T4 pe yeh support nahi karta.

**Fix:** `megablox=False` + `sparse_matmul=False` — dense matmul fallback use karo.
But T4 pe Triton issue bhi tha isliye dono fixes simultaneously chahiye the.

**Concept:** MegaBlox ek specialized sparse matrix multiplication kernel hai
jo MoE ke expert routing ko efficiently compute karta hai. T4 pe nahi chala.

---

## TPU Backend (v5e-1) — Single Process Lock Problem

```
ABORTED: The TPU is already in use by process with pid 27846.
```

**Kya hua:** Yeh sabse interesting failure hai.

Colab notebook ek Python kernel process mein run hota hai. Jab tune
`import jax` karke verification cell run kiya, JAX ne TPU ko us process mein initialize kar liya.
TPU v5e-1 ek time pe sirf EK process use kar sakta hai.

Phir jab `run_train()` ne `python3 -m maxtext.trainers.pre_train.train` subprocess launch kiya,
woh subprocess ek ALAG process tha (alag PID). TPU already kernel process ke paas tha,
isliye subprocess ko `ABORTED` mila.

**Fix jo try kiya:** In-process execution — MaxText ko subprocess ki jagah
seedha Python mein call karo. Lekin tab `No config file provided` error aaya
kyunki absl (Google ka config library) ka CLI entry point properly initialize nahi hua.

**Root cause:** Colab ka free-tier TPU (v5e-1) single-chip hai, single-process hai.
Production pe GKE TPU pods use hote hain jo multi-process distributed training support karte hain.

---

## Architecture Concepts — Discussion ke liye

### Qwen3 Architecture (dense model)

```
model_name: qwen3-0.6b
base_emb_dim: 1024        ← embedding dimension (hidden size)
base_num_query_heads: 16  ← multi-head attention heads
base_num_kv_heads: 8      ← GQA (Grouped Query Attention) — fewer KV heads
base_mlp_dim: 3072        ← feed-forward network size
base_num_decoder_layers: 28 ← number of transformer layers
head_dim: 128             ← per-head dimension
```

**1.7B scaling ke liye kya change kiya:**
```
base_emb_dim: 1024 → 2048   (2x embedding = ~4x param increase)
base_mlp_dim: 3072 → 5504   (larger FFN)
head_dim: 64 → 128          (larger per-head attention)
```
`override_model_config=True` isliye chahiye tha kyunki MaxText ke YAML config mein
ye values hardcoded hain — CLI se override karne ke liye yeh flag lagana padta hai.

**GQA (Grouped Query Attention):** Standard attention mein Q, K, V sab same number of heads hote hain.
GQA mein K aur V ke fewer heads hote hain (8 vs 16) — memory efficient hota hai
kyunki KV cache chhota hota hai inference mein.

### DeepSeek MoE Architecture

**MoE (Mixture of Experts) kya hai:**
Normal transformer mein har token same FFN layers se guzarta hai.
MoE mein multiple "expert" FFN layers hote hain. Router decide karta hai
ki har token ke liye kaun se experts activate hon.

```
num_experts: 8        ← total 8 expert FFN networks
num_experts_per_tok: 2 ← har token sirf 2 experts use karta hai
shared_experts: 1      ← 1 expert har token pe always active
```

**Active vs Total Parameters:**
- Total params: 0.269B (saare 8 experts + attention)
- Active params: 0.153B (sirf 2 routed + 1 shared expert per token)

Yeh MoE ka main advantage hai — model bada hota hai (zyada knowledge)
lekin compute chhota hota hai (sirf kuch experts active per token).

**Sub-1B banane ke liye kya shrink kiya (original DeepSeek-V2 16B se):**

| Parameter | Original | Humara | Reason |
|---|---|---|---|
| `base_emb_dim` | 2048 | 768 | Sabse bada size driver |
| `base_num_decoder_layers` | ~27 | 12 | Layers kam = params drastically kam |
| `num_experts` | 64+ | 8 | Expert count biggest MoE param |
| `base_moe_mlp_dim` | 1536+ | 768 | Expert FFN width |

**MLA (Multi-head Latent Attention) — DeepSeek ka special attention:**
Normal attention mein K aur V directly store hote hain.
MLA mein low-rank compressed representation store hoti hai:
```
q_lora_rank: 192   ← query low-rank projection
kv_lora_rank: 48   ← key-value low-rank projection
```
Yeh KV cache ko drastically reduce karta hai inference mein.

**MLA constraints jo failures cause kiye:**
```
num_query_heads == num_kv_heads  (MLA mein GQA nahi hoti)
v_head_dim == head_dim == qk_nope_head_dim + qk_rope_head_dim
              = 64 + 32 = 96
```

---

## Metrics Samajhna

| Metric | Kya matlab hai |
|---|---|
| `seconds` | Ek training step kitne seconds mein complete hua |
| `TFLOP/s/device` | Tera-floating-point operations per second per device — GPU utilization |
| `Tokens/s/device` | Kitne tokens per second process hue — throughput |
| `loss` | Model ka training loss — kam hona chahiye |
| `grad_norm` | Gradient ka magnitude — agar bahut bada ho to training unstable |
| `learning_rate` | Cosine decay schedule follow karta hai |
| `mem_total_gb` | Compile time pe estimated total memory usage |

**CPU vs GPU vs TPU comparison (agar kisi ne successfully run kiya hota):**
- CPU: Slow — no hardware acceleration, general purpose cores
- GPU: Fast — parallel matrix multiplication, dedicated VRAM
- TPU: Fastest — specifically designed for tensor operations, HBM bandwidth

---

## Discussion mein kya pooch sakte hain — aur short answers

**Q: Synthetic data kab use karte hain real training mein?**
A: Kabhi nahi. Sirf benchmarking ke liye. Real training mein Grain + ArrayRecord use hota hai.

**Q: MoE aur dense model mein kya tradeoff hai?**
A: MoE: zyada total params (more knowledge) lekin same compute (only top-k experts active).
Dense: simple, sab params har token pe active, easier to train aur deploy.

**Q: T4 pe 1.7B kyu nahi chala?**
A: 1.7B ko weights + gradients + AdamW state ~19GB chahiye. T4 VRAM 15GB hai.

**Q: TPU pe kyu fail hua?**
A: Colab v5e-1 single-process hai. Notebook kernel ne TPU claim kar liya,
subprocess usse acquire nahi kar saka.

**Q: `attention=dot_product` kyun add kiya?**
A: CPU aur T4 pe MaxText GPU Pallas attention kernel use karta hai by default.
CPU pe Pallas nahi chalta, T4 pe Triton needs Ampere (8.0+).
`dot_product` pure JAX implementation hai jo sab jagah chalta hai.

**Q: `override_model_config=True` kya karta hai?**
A: MaxText har model ka ek YAML config file hota hai. CLI se individual params override
karne ke liye yeh flag lagana padta hai, warna CLI values ignore ho jaati hain.

---

## Summary table — poori assignment

| | CPU | GPU T4 | TPU v5e-1 |
|---|---|---|---|
| Qwen 0.6B | ❌ OOM (11.4GB needed) | ✅ **2.48 TFLOP/s** | ❌ Process lock |
| Qwen 1.7B | ❌ OOM | ❌ OOM (19GB needed) | ❌ Process lock |
| DeepSeek MoE | ❌ Pallas CPU limit | ❌ T4 Triton limit | ❌ Process lock |

**Ek successful run: Qwen 0.6B on GPU T4**
- 50 steps, 104 seconds, 2.48 TFLOP/s, 661 tokens/s, 10.8 GB memory

---

> Yeh doc sirf understanding ke liye hai.
> README apne words mein likhna — yahi tumhara actual work hai.
