import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.naive import NaiveEngine
from inferlab.engines.speculative import SpeculativeEngine
from inferlab.engines.base import GenerationConfig

draft_name = "Qwen/Qwen2.5-0.5B"
target_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(target_name)

prompt = "The quick brown fox jumps over the lazy dog and then"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
config = GenerationConfig(max_new_tokens=30, do_sample=False, eos_token_id=tokenizer.eos_token_id)

# Load each model ONCE, share where safe
target_model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16).to("cuda")
draft_model = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=torch.float16).to("cuda")

naive_engine = NaiveEngine(target_model, device="cuda")
naive_result = naive_engine.generate(input_ids, config)
print(f"[naive] first token: {naive_result.generated_ids[0][0].item()}, decoded: {tokenizer.decode(naive_result.generated_ids[0][0:1])}")

spec_engine = SpeculativeEngine(draft_model, target_model, device="cuda", num_draft_tokens=4)
spec_result = spec_engine.generate(input_ids, config, tokenizer=tokenizer, naive_result=naive_result)

print("Naive:      ", tokenizer.decode(naive_result.generated_ids[0]))
print("Speculative:", tokenizer.decode(spec_result.generated_ids[0]))
print("Match:", torch.equal(naive_result.generated_ids, spec_result.generated_ids))
print("Acceptance rate:", spec_result.extra["acceptance_rate"])
print("Total accepted / drafted:", spec_result.extra["total_accepted"], "/", spec_result.extra["total_drafted"])
