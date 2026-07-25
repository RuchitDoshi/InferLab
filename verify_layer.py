import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.attention.from_scratch_layer import qwen_full_forward

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cuda")
model.eval()
print(model.lm_head.weight is model.model.embed_tokens.weight)

prompt = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

prefill_ids = input_ids[:, :-1]
new_token_id = input_ids[:, -1:]

with torch.no_grad():
    # Reference: full sequence, no cache, take last position only
    hf_full_output = model.model(input_ids).last_hidden_state
    hf_new_reference = hf_full_output[:, -1:, :]

    # From-scratch cached path
    from inferlab.engines.kv_cache import KVCache
    my_cache = KVCache()
    _ = qwen_full_forward(prefill_ids, model, kv_cache=my_cache)
    scratch_new = qwen_full_forward(new_token_id, model, kv_cache=my_cache)

diff = (hf_new_reference - scratch_new).abs().max().item()
print(f"Full model cached-decode max abs diff: {diff:.8f}")


