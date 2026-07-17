import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.naive import NaiveEngine
from inferlab.engines.kv_cache import KVCacheEngine
from inferlab.engines.from_scratch import FromScratchEngine
from inferlab.engines.base import GenerationConfig
from inferlab.eval.prompts import load_benchmark_prompts
from inferlab.bench.runner import run_sweep

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_naive = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model_kv = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model_scratch = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

naive_engine = NaiveEngine(model_naive, device="cuda")
kv_engine = KVCacheEngine(model_kv, device="cuda")
scratch_engine = FromScratchEngine(model_scratch, device="cuda")

engines = {"naive": naive_engine, "kv_cache": kv_engine, "from_scratch": scratch_engine}

prompt_tiers = load_benchmark_prompts(tokenizer, num_prompts_per_tier=3)

config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

rows = run_sweep(engines, prompt_tiers, config, output_path="results/week1_sweep_v2.csv")

import pandas as pd
df = pd.DataFrame(rows)
print(df[["engine", "tier", "prompt_idx", "ttft_ms", "tpot_ms", "tokens_per_sec", "peak_vram_mb", "status"]])