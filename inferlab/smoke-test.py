import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.from_scratch import FromScratchEngine
from inferlab.engines.base import GenerationConfig
from inferlab.eval.prompts import load_benchmark_prompts
from inferlab.bench.runner import run_sweep

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_contig = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model_paged = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

contig_engine = FromScratchEngine(model_contig, device="cuda", cache_type="kv_cache")
paged_engine = FromScratchEngine(model_paged, device="cuda", cache_type="paged_kv_cache")

engines = {"from_scratch_contig": contig_engine, "from_scratch_paged": paged_engine}

prompt_tiers = load_benchmark_prompts(tokenizer, num_prompts_per_tier=3)
config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

rows = run_sweep(engines, prompt_tiers, config, output_path="results/paged_vs_contig.csv")

import pandas as pd
df = pd.DataFrame(rows)
print(df[["engine", "tier", "prompt_idx", "ttft_ms", "tpot_ms", "tokens_per_sec", "peak_vram_mb", "status"]])