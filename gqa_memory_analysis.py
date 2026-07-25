"""
GQA memory analysis: validates KVCache.memory_bytes() (measured) against
theoretical_kv_cache_bytes() (calculated), and quantifies GQA's memory
savings vs. a hypothetical plain-MHA config at the same model size.

Run: python gqa_memory_analysis.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferlab.engines.from_scratch import FromScratchEngine
from inferlab.engines.base import GenerationConfig
from inferlab.metrics.memory import theoretical_kv_cache_bytes

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

prompt = "The quick brown fox jumps over the lazy dog and then"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
prompt_len = input_ids.shape[1]

engine = FromScratchEngine(model, device="cuda")
config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

result = engine.generate(input_ids, config)

measured_bytes = engine.cache.memory_bytes()
generated_len = result.generated_ids.shape[1]
full_seq_len = prompt_len + generated_len - 1  # -1 because the last token is EOS and not cached

theoretical_bytes = theoretical_kv_cache_bytes(
    num_layers=24, num_kv_heads=2, head_dim=64,
    seq_len=full_seq_len, dtype_bytes=2, batch_size=1,
)

print(f"Prompt length: {prompt_len}, generated: {generated_len}, total seq_len: {full_seq_len}")
print(f"Measured cache bytes:    {measured_bytes:,}")
print(f"Theoretical cache bytes: {theoretical_bytes:,}")
print(f"Match: {measured_bytes == theoretical_bytes}")

# Hypothetical: what if this model used plain MHA (num_kv_heads == num_attention_heads)
# instead of GQA, at the same hidden_size/num_layers/seq_len?
hypothetical_mha_bytes = theoretical_kv_cache_bytes(
    num_layers=24, num_kv_heads=14, head_dim=64,  # 14 = num_attention_heads, i.e. no GQA
    seq_len=full_seq_len, dtype_bytes=2, batch_size=1,
)

savings_ratio = hypothetical_mha_bytes / theoretical_bytes
savings_pct = (1 - theoretical_bytes / hypothetical_mha_bytes) * 100

print(f"\nHypothetical MHA (14 KV heads) cache bytes: {hypothetical_mha_bytes:,}")
print(f"Actual GQA (2 KV heads) cache bytes:         {theoretical_bytes:,}")
print(f"GQA memory savings: {savings_ratio:.1f}x smaller ({savings_pct:.1f}% reduction)")