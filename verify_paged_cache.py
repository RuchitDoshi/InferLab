import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.attention.from_scratch_layer import qwen_full_forward
from inferlab.engines.kv_cache import KVCache
from inferlab.attention.paged_kv_cache import PagedKVCache

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cuda")
model.eval()

prompt = "The quick brown fox jumps over the lazy dog and then"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

prefill_ids = input_ids[:, :-1]
new_token_id = input_ids[:, -1:]

config = model.config
num_layers = config.num_hidden_layers
num_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads

with torch.no_grad():
    contig_cache = KVCache()
    contig_out_prefill = qwen_full_forward(prefill_ids, model, kv_cache=contig_cache)
    contig_out_new = qwen_full_forward(new_token_id, model, kv_cache=contig_cache)

    paged_cache = PagedKVCache(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                 head_dim=head_dim, block_size=16, num_blocks=100,
                                 dtype=torch.float32, device="cuda")
    paged_out_prefill = qwen_full_forward(prefill_ids, model, kv_cache=paged_cache)
    paged_out_new = qwen_full_forward(new_token_id, model, kv_cache=paged_cache)

diff_prefill = (contig_out_prefill - paged_out_prefill).abs().max().item()
diff_new = (contig_out_new - paged_out_new).abs().max().item()
print(f"Prefill max diff (paged vs contiguous): {diff_prefill:.8f}")
print(f"Decode max diff (paged vs contiguous): {diff_new:.8f}")