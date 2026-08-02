"""
Acceptance-rate-vs-speedup benchmark: SpeculativeEngine at several
num_draft_tokens (K) values, compared against NaiveEngine (target model
alone) as the baseline it's meant to speed up.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferlab.engines.naive import NaiveEngine
from inferlab.engines.speculative import SpeculativeEngine
from inferlab.engines.base import GenerationConfig
from inferlab.eval.prompts import load_benchmark_prompts
from inferlab.bench.runner import run_sweep

draft_name = "Qwen/Qwen2.5-0.5B"
target_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(target_name)

target_model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16).to("cuda")
draft_model = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=torch.float16).to("cuda")

engines = {
    "naive_target_only": NaiveEngine(target_model, device="cuda"),
    "spec_k2": SpeculativeEngine(draft_model, target_model, device="cuda", num_draft_tokens=2),
    "spec_k4": SpeculativeEngine(draft_model, target_model, device="cuda", num_draft_tokens=4),
    "spec_k8": SpeculativeEngine(draft_model, target_model, device="cuda", num_draft_tokens=8),
}

prompt_tiers = load_benchmark_prompts(tokenizer, num_prompts_per_tier=3)
config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

rows = run_sweep(engines, prompt_tiers, config, output_path="results/speculative_sweep.csv")

import pandas as pd
df = pd.DataFrame(rows)
print(df[["engine", "tier", "prompt_idx", "ttft_ms", "tpot_ms", "tokens_per_sec", "acceptance_rate", "status"]])