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
    TODO(ruchit):
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
    raise NotImplementedError