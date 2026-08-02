"""
Quality benchmark: perplexity of fp16 vs int8 (bitsandbytes) on the SAME
fixed benchmark prompts, across all three tiers -- measures how much
quantization degrades fluency/confidence relative to fp16, not just
speed/memory.

Run:
    python bench_quantization_quality.py --precision fp16
    python bench_quantization_quality.py --precision int8
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inferlab.eval.prompts import load_benchmark_prompts
from inferlab.eval.quality import compute_perplexity_windowed


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

    prompt_tiers = load_benchmark_prompts(tokenizer, num_prompts_per_tier=3)

    for tier_name, tier in prompt_tiers.items():
        for i, prompt_ids in enumerate(tier.prompts):
            text = tokenizer.decode(prompt_ids[0])
            ppl = compute_perplexity_windowed(model, tokenizer, text, device="cuda")
            print(f"[{args.precision}] tier={tier_name} prompt_idx={i} perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()