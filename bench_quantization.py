"""
Quantization benchmark: fp16 vs int8 (bitsandbytes) for Qwen2.5-1.5B
through KVCacheEngine. Run as two SEPARATE process invocations (like
bench_scheduler.py) so each measurement is clean -- loading both models
in one process leaves the first model's weights resident on the GPU,
polluting the second condition's peak memory measurement.

Run:
    python bench_quantization.py --precision fp16
    python bench_quantization.py --precision int8
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inferlab.engines.kv_cache import KVCacheEngine
from inferlab.engines.base import GenerationConfig
from inferlab.eval.prompts import load_benchmark_prompts
from inferlab.bench.runner import run_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    args = parser.parse_args()

    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cuda")

    engines = {args.precision: KVCacheEngine(model, device="cuda")}

    prompt_tiers = load_benchmark_prompts(tokenizer, num_prompts_per_tier=3)
    config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

    rows = run_sweep(engines, prompt_tiers, config, output_path=f"results/quantization_{args.precision}.csv")

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["engine", "tier", "prompt_idx", "ttft_ms", "tpot_ms", "tokens_per_sec", "peak_vram_mb", "status"]])


if __name__ == "__main__":
    main()