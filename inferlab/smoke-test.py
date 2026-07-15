# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from inferlab.engines.naive import NaiveEngine
# from inferlab.engines.kv_cache import KVCacheEngine
# from inferlab.engines.base import GenerationConfig

# model_name = "sshleifer/tiny-gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)


# print("Testing NaiveEngine...\n\n")
# engine = NaiveEngine(model, device="cpu")  # CPU is fine for a tiny model

# prompt = "The quick brown fox"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# config = GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=tokenizer.eos_token_id)

# result = engine.generate(input_ids, config)

# print("Generated ids:", result.generated_ids)
# print("Decoded:", tokenizer.decode(result.generated_ids[0]))
# print("Full output:", tokenizer.decode(result.output_ids[0]))
# print("Num timestamps:", len(result.step_timestamps))
# print("Num generated tokens:", result.generated_ids.shape[1])


# print("\n\nNow testing KVCacheEngine...\n\n")

# engine = KVCacheEngine(model, device="cpu")  # CPU is fine for a tiny model

# prompt = "The quick brown fox"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# config = GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=tokenizer.eos_token_id)

# result = engine.generate(input_ids, config)

# print("Generated ids:", result.generated_ids)
# print("Decoded:", tokenizer.decode(result.generated_ids[0]))
# print("Full output:", tokenizer.decode(result.output_ids[0]))
# print("Num timestamps:", len(result.step_timestamps))
# print("Num generated tokens:", result.generated_ids.shape[1])


import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.naive import NaiveEngine
from inferlab.engines.kv_cache import KVCacheEngine
from inferlab.engines.base import GenerationConfig

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_naive = AutoModelForCausalLM.from_pretrained(model_name)
model_kv = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The quick brown fox"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
config = GenerationConfig(max_new_tokens=100, do_sample=False, eos_token_id=tokenizer.eos_token_id)

naive_engine = NaiveEngine(model_naive, device="cpu")
kv_engine = KVCacheEngine(model_kv, device="cpu")

t0 = time.perf_counter()
naive_result = naive_engine.generate(input_ids, config)
t1 = time.perf_counter()
print(f"Naive: {t1 - t0:.3f}s for {naive_result.generated_ids.shape[1]} tokens")

t0 = time.perf_counter()
kv_result = kv_engine.generate(input_ids, config)
t1 = time.perf_counter()
print(f"KV Cache: {t1 - t0:.3f}s for {kv_result.generated_ids.shape[1]} tokens")

print("Match:", torch.equal(naive_result.generated_ids, kv_result.generated_ids))