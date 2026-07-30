"""
Multi-sequence scheduler benchmark: shared PagedKVCache (ContinuousBatchScheduler)
vs private per-sequence KVCache (PrivateCacheScheduler). Only the scheduler
class differs between runs -- everything else (prompts, max_active, model,
metrics) is identical, to isolate the shared-pool-vs-private-pools variable.

Run:
    python bench_scheduler.py --scheduler shared
    python bench_scheduler.py --scheduler private
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferlab.engines.continuous_batch import ContinuousBatchScheduler
from inferlab.engines.continuous_batch import PrivateCacheScheduler
from inferlab.eval.prompts import load_benchmark_prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", choices=["shared", "private"], required=True)
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--max_active", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args = parser.parse_args()

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    prompts = load_benchmark_prompts(tokenizer, num_prompts_per_tier=5)
    prompts_list = prompts["short"].prompts + prompts["medium"].prompts
    prompts_list = [i.to(model.device) for i in prompts_list]


    if args.scheduler == "shared":
        scheduler = ContinuousBatchScheduler(model, max_active=args.max_active)
    else:
        scheduler = PrivateCacheScheduler(model, max_active=args.max_active)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for prompt_ids in prompts_list:
        scheduler.submit(prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    finished = scheduler.run()

    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    result = {
        "scheduler": args.scheduler,
        "num_sequences": args.num_sequences,
        "max_active": args.max_active,
        "elapsed_sec": elapsed,
        "peak_allocated_mb": peak_allocated / (1024 * 1024),
        "peak_reserved_mb": peak_reserved / (1024 * 1024),
        "fragmentation_gap_mb": (peak_reserved - peak_allocated) / (1024 * 1024),
    }
    print(result)

    with open(f"results/bench_{args.scheduler}.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()