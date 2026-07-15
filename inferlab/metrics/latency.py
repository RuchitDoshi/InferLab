"""
Derive latency metrics from a GenerationResult.step_timestamps.

step_timestamps convention (set this in Engine.generate()):
    step_timestamps[0]  = time right after the 1st decode_step() returns
    step_timestamps[1]  = time right after the 2nd decode_step() returns
    ...
    len(step_timestamps) == num_generated_tokens

Also needs a reference "generate_start" timestamp captured right before
prefill() is called -- TODO(ruchit): decide whether that lives as
step_timestamps[-1]-adjacent metadata on GenerationResult.extra, or as a
separate argument passed into these functions. Pick one, document it in
base.py's GenerationResult docstring, and be consistent.
"""

from dataclasses import dataclass


@dataclass
class LatencyReport:
    ttft_ms: float          # time to first token = prefill duration
    tpot_ms: float          # mean time per output token (decode steps only)
    e2e_ms: float            # total wall clock, prefill + all decode steps
    per_token_ms: list       # raw per-decode-step latencies, for plotting
                              # distributions / tail latency (p50/p99) later


def compute_latency(step_timestamps: list, generate_start: float) -> LatencyReport:
    """
    - ttft_ms = (step_timestamps[0] - generate_start) * 1000
    - per_token_ms = differences between consecutive step_timestamps[1:]
      (each decode step's individual duration)
    - tpot_ms = mean(per_token_ms)
    - e2e_ms = (step_timestamps[-1] - generate_start) * 1000

    EDGE CASE: what if only 1 token was generated (step_timestamps has just
    the prefill entry, no decode steps)? tpot_ms and per_token_ms should
    degrade gracefully (empty list / None), not raise.
    """
    ttft_ms = (step_timestamps[0] - generate_start) * 1000
    per_token_ms = [(step_timestamps[i] - step_timestamps[i - 1]) * 1000 for i in range(1, len(step_timestamps))]
    tpot_ms = sum(per_token_ms) / len(per_token_ms) if per_token_ms else None
    e2e_ms = (step_timestamps[-1] - generate_start) * 1000

    return LatencyReport(ttft_ms=ttft_ms, tpot_ms=tpot_ms, e2e_ms=e2e_ms, per_token_ms=per_token_ms)