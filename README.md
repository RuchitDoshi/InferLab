# InferLab# inferlab

From-scratch LLM inference optimization techniques — KV caching, PagedAttention, continuous batching, speculative decoding, and quantization — implemented by hand, verified against reference implementations, and benchmarked honestly.

This project exists to demonstrate understanding of *how* modern inference engines (vLLM, TensorRT-LLM) work internally, not to compete with them. Every technique here is a simplified, from-scratch reimplementation for learning and demonstration — not a production-grade alternative.

**Models**: Qwen2.5-0.5B (24 layers) and Qwen2.5-1.5B (28 layers, used as a speculative decoding target and quantization target).
**Hardware used for all benchmarks below**: NVIDIA RTX 3070, 8GB VRAM.

---

## Why this project

Most "inference optimization" portfolio projects wrap an existing engine or implement one technique in isolation. This project instead:

- Implements every technique **from raw PyTorch tensor ops** — no calls into HuggingFace's own forward pass anywhere in the core engines.
- **Verifies correctness before trusting performance**, at every stage — single layer, cached layer, full model, end-to-end generation — against HuggingFace's real implementation.
- **Reports honest, multi-sided findings**, not headline numbers. Every technique below has a real tradeoff, and this README states it plainly.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.10, PyTorch ≥2.2, a CUDA GPU for anything beyond the correctness test suite.

---

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.from_scratch import FromScratchEngine
from inferlab.engines.base import GenerationConfig

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

engine = FromScratchEngine(model, device="cuda", cache_type="kv_cache")
input_ids = tokenizer("The quick brown fox", return_tensors="pt").input_ids.to("cuda")
config = GenerationConfig(max_new_tokens=30, do_sample=False, eos_token_id=tokenizer.eos_token_id)

result = engine.generate(input_ids, config)
print(tokenizer.decode(result.generated_ids[0]))
```

Run the test suite:

```bash
pytest tests/test_engines.py -v
```

---

## Techniques & results

### 1. Naive baseline + KV caching

Naive re-runs the full sequence at every decode step. KV caching stores each layer's key/value tensors so only the newest token needs processing.

| Tier (context) | Naive (ms/token) | KV cache (ms/token) |
|---|---|---|
| short (128) | ~20 | ~18 |
| medium (1024) | ~42 | ~18 |
| long (4096) | ~169 | ~21 |

**~8x faster at long context.** Counterintuitive finding: naive uses *more* peak VRAM than the cached version at every tier — its transient full-sequence attention matrices are larger than what the cache actually stores.

### 2. From-scratch Qwen2.5 forward pass

RMSNorm, RoPE, GQA-aware causal attention, SwiGLU MLP — reimplemented from raw weight tensors, verified numerically identical to HuggingFace's real forward pass at every level: single layer, cached layer, full model, cached full model, end-to-end generation (exact token match vs. the naive baseline).

Overhead vs. HuggingFace's own cache concentrates almost entirely in **prefill** (~7-8x gap — fused kernels matter most on large batched matmuls), not decode (~1.3x gap — small per-token steps, where fusion matters less).

### 3. GQA memory analysis

| Config | KV heads | Relative cache memory |
|---|---|---|
| Plain MHA (hypothetical) | 14 | 7.0x |
| GQA (actual, Qwen2.5-0.5B) | 2 | 1.0x |

**Derived, verified 7.0x KV-cache memory reduction** from GQA vs. a plain-MHA config at the same model size.

### 4. Simplified PagedAttention

Fixed-size memory blocks from a shared pool instead of contiguous reallocate-and-copy growth — the core idea behind vLLM's PagedAttention.

**Single-sequence**, after two rounds of profiling-driven optimization:

| Version | Medium-tier TTFT | vs. contiguous cache |
|---|---|---|
| V1 (naive block reconstruction) | 4837ms | ~180x slower |
| V2 (batched allocation) | 825ms | ~30x slower |
| V3 (vectorized block writes) | 86ms | ~3.2x slower |

Final honest residual gap (~1.4-3.3x across tiers) is a real, structural cost — gathering scattered blocks is inherently more expensive than one contiguous tensor's append — not remaining implementation inefficiency.

**Multi-sequence** (10 concurrent sequences, shared pool vs. private per-sequence caches, single run):

| Metric | Shared pool | Private caches |
|---|---|---|
| Fragmentation waste (reserved − allocated) | 24.5 MB | 50.1 MB |
| Wall-clock time | 8.46s | 7.11s |

Shared pooling cuts fragmentation ~2x but costs ~19% throughput — **a genuine tradeoff, not a clean win**. Single-sequence benchmarks alone showed paging as strictly worse everywhere; this required building real multi-sequence contention to reveal the actual benefit.

### 5. Continuous batching

An admit/decode/evict scheduler over a shared `PagedKVCache` pool — verified correct under deliberately forced admit/evict cycling (more waiting requests than active slots); every sequence's output is identical to running it alone. This is the scheduling layer that makes the multi-sequence PagedAttention benchmark above possible at all.

### 6. Speculative decoding

A small draft model (Qwen2.5-0.5B) proposes K candidate tokens per round; the large target model (Qwen2.5-1.5B) verifies all K in one batched forward pass instead of paying its full cost per token.

| Engine | Short tok/s | Medium tok/s | Long tok/s |
|---|---|---|---|
| Target alone (naive) | ~33-40 | ~8.8 | ~2.3 |
| Speculative, K=4 | ~13-18 | ~11.3-13.0 | ~2.4-3.2 |
| Speculative, K=8 | ~9.6-16.2 | ~8.7-10.8 | ~3.3-3.6 |

**Speculative decoding is slower than the target model alone at short/medium context**, and only reaches **~1.5x speedup at long context** — where the target's own per-token cost is expensive enough to be worth amortizing. Acceptance rate drops as K increases (0.19-0.71 depending on tier and K), matching theory: longer unverified guesses compound uncertainty.

Verified exactly correct (token-for-token identical to the target model alone) after finding and fixing five distinct bugs: two fp16 numerical precision failures (attention-score overflow producing NaN via softmax's internal `Inf − Inf`; RMSNorm underflow/"swamping" in a persistent residual-stream outlier value), a shift-by-one indexing error in the accept/reject comparison, a missing cache update for the first committed token, and a causal-mask bug specific to the one input shape (multiple new tokens against a non-empty cache) that no other technique in this project ever produces.

### 7. Quantization (int8, `bitsandbytes`)

Wrapped, not built from scratch — quantized weight storage is incompatible with the from-scratch layer's direct tensor access, so this runs through `KVCacheEngine` only.

| Metric | fp16 | int8 | Change |
|---|---|---|---|
| Peak VRAM (short/medium/long) | 2995 / 3281 / 4265 MB | 1752 / 2038 / 3024 MB | ~30-41% less |
| Latency (ms/token) | ~22-29 | ~106-113 | **~4-5x slower** |
| Perplexity (9 test prompts) | baseline | +0.1% to +4.3% | small, real degradation |

Real memory savings and modest quality cost (int8's reputation as a "safe" quantization level holds up) — but on this hardware (no native int8 tensor-core path), the **latency cost is large enough to outweigh the memory benefit** for many use cases. Stated plainly rather than only touting the memory win.

---

## Project structure
inferlab/
├── engines/ naive, kv_cache, from_scratch, speculative, continuous_batch
├── attention/ from-scratch layer forward pass, paged KV cache
├── metrics/ latency, throughput, memory
├── eval/ benchmark prompts, quality (perplexity)
├── bench/ sweep runner
tests/ correctness suite (5 tests, all engines vs. naive reference)
bench_.py standalone benchmark scripts (repo root)
verify_.py standalone correctness verification scripts (repo root)

## Known limitations (stated explicitly, not hidden)

- Sequences are processed via separate forward calls per sequence in the continuous batching scheduler, not true batched multi-sequence attention (would require custom kernels — out of scope for a PyTorch-level reimplementation).
- PagedAttention here is single-tenant memory management; no cross-sequence prefix sharing (a real vLLM feature).
- Speculative decoding's draft cache is rebuilt from scratch every round rather than incrementally rolled back — a deliberate simplicity tradeoff, and the direct cause of its short-context slowdown.
- No FlashAttention-style kernel fusion — this project focuses on caching/memory/scheduling strategies, not compute-kernel optimization.
