import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.from_scratch import FromScratchEngine
from inferlab.engines.base import GenerationConfig
from inferlab.engines.continuous_batch import ContinuousBatchScheduler

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Reference: solo runs via FromScratchEngine (paged backend) ---
prompts = [
    "The quick brown fox jumps over the lazy dog",
    "Once upon a time in a land far away",
    "Machine learning is a subfield of artificial intelligence",
]
max_new_tokens = 15

solo_results = {}
for i, prompt in enumerate(prompts):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    engine = FromScratchEngine(model, device="cuda", cache_type="paged_kv_cache")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    result = engine.generate(input_ids, config)
    solo_results[i] = result.generated_ids

# --- Scheduler: same 3 prompts, run CONCURRENTLY ---
model_sched = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
scheduler = ContinuousBatchScheduler(model_sched, max_active=2)  # deliberately < 3, forces admit/evict cycling

sequence_id_to_prompt_idx = {}
for i, prompt in enumerate(prompts):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    seq_id = scheduler.submit(input_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    sequence_id_to_prompt_idx[seq_id] = i

finished = scheduler.run()

# --- Compare ---
for seq_id, state in finished.items():
    prompt_idx = sequence_id_to_prompt_idx[seq_id]
    match = torch.equal(solo_results[prompt_idx], state.generated_ids())
    print(f"Prompt {prompt_idx}: match = {match}")
    if not match:
        print(f"  solo:      {solo_results[prompt_idx]}")
        print(f"  scheduler: {state.generated_ids()}")