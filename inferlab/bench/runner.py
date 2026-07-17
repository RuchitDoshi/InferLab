"""
Sweep runner: given a list of engines and the prompt tiers, run
generate() on each (engine x tier x prompt), collect latency/throughput/
memory/quality metrics, and dump to CSV/JSON for report.py to consume.

This is the Week 1 payoff -- once this works for {naive, kv_cache}, every
later engine just plugs in with zero changes to this file. Resist adding
engine-specific branches here; if you feel the urge, it's a sign your
Engine interface (base.py) needs a more general hook instead.
"""

import csv
import time
from pathlib import Path

import torch
import pandas as pd

from inferlab.engines.base import Engine, GenerationConfig
from inferlab.metrics.latency import compute_latency
from inferlab.metrics.throughput import compute_throughput
from inferlab.metrics.memory import peak_vram_bytes


def run_sweep(
    engines: dict[str, Engine],
    prompt_tiers: dict,  # name -> PromptTier
    config: GenerationConfig,
    output_path: str = "results/week1_sweep.csv",
    device: str = "cuda",
) -> list[dict]:
    """
    For each (engine_name, engine) x (tier_name, tier) x prompt in tier.prompts:
        1. torch.cuda.reset_peak_memory_stats(device)
        2. generate_start = time.perf_counter()
        3. result = engine.generate(input_ids, config)
        4. latency = compute_latency(result.step_timestamps, generate_start)
        5. throughput = compute_throughput(len(result.generated_ids[0]), ...)
        6. mem = peak_vram_bytes(device)
        7. append a row dict: {engine, tier, prompt_idx, ttft_ms, tpot_ms,
           e2e_ms, tokens_per_sec, peak_vram_mb, ...}
    Write all rows to output_path as CSV (Path(output_path).parent.mkdir
    (parents=True, exist_ok=True) first). Return the list of row dicts too,
    so report.py / a notebook can consume it directly without re-reading
    the CSV.

    EDGE CASE: one engine crashing on one prompt (e.g. OOM on the long
    tier) shouldn't kill the whole sweep -- catch, log the failure into the
    row (e.g. status="OOM"), and continue. You want a full comparison table
    even if one cell is a documented failure; that's a legitimate finding,
    not a bug to hide.
    """
    rows = []

    for engine_name, engine in engines.items():
        for tier_name, tier in prompt_tiers.items():
            for i, input_ids in enumerate(tier.prompts):

                try:
                    
                    # Move input_ids to the correct device
                    input_ids = input_ids.to(device)
                
                    # Reset peak memory stats and start timing
                    torch.cuda.reset_peak_memory_stats(device)
                    
                    # Run the engine's generate method
                    result = engine.generate(input_ids, config)

                    # Compute metrics
                    latency = compute_latency(result.step_timestamps, result.generate_start)
                    e2e_seconds = latency.e2e_ms / 1000
                    throughput = compute_throughput(len(result.generated_ids[0]), e2e_seconds)
                    mem = peak_vram_bytes(device)

                    # Append row
                    rows.append({
                        "engine": engine_name,
                        "tier": tier_name,
                        "prompt_idx": i,
                        "ttft_ms": latency.ttft_ms,
                        "tpot_ms": latency.tpot_ms,
                        "e2e_ms": latency.e2e_ms,
                        "per_token_ms": latency.per_token_ms,
                        "tokens_per_sec": throughput.tokens_per_sec,
                        "requests_per_sec": throughput.requests_per_sec,
                        "peak_vram_mb": mem / (1024 * 1024),
                        "status": "SUCCESS"
                    })

                except Exception as e:
                    rows.append({"engine": engine_name, "tier": tier_name, "prompt_idx": i, "status": f"FAILED: {e}"})


    # Write to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return rows
