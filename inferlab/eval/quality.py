"""
Quality-regression harness: perplexity + exact-match, run for every engine
against a fixed prompt set, diffed against the fp16 NaiveEngine baseline.

This gets used starting Week 1 (even though naive vs kv_cache SHOULD show
zero quality delta -- that's the point, it's a correctness signal as much
as a quality one) and becomes essential in Week 3-4 for speculative
decoding and quantization, where nonzero deltas are expected and need to
be characterized, not just detected.
"""

from dataclasses import dataclass

import torch


@dataclass
class QualityReport:
    perplexity: float
    exact_match_rate: float  # fraction of prompts where greedy output
                              # matches the baseline engine's output exactly


def compute_perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    """
    TODO(ruchit): standard perplexity computation --
    1. Tokenize text, move to device.
    2. Forward pass with labels=input_ids (HF models return .loss as mean
       cross-entropy over non-masked tokens when you pass labels).
    3. perplexity = exp(loss).

    EDGE CASE: very long text may need to be chunked (stride/window) to fit
    memory on a 3070 -- decide your max sequence length for eval prompts now
    (tie this to the "long-context tier" prompt set) so you don't hit OOM
    mid-benchmark-run and lose a sweep.
    """
    raise NotImplementedError


def exact_match(reference_ids: torch.Tensor, candidate_ids: torch.Tensor) -> bool:
    """
    TODO(ruchit): True if candidate_ids equals reference_ids exactly
    (same length, same tokens). Used to compare an engine's greedy output
    against NaiveEngine's greedy output on the same prompt.

    EDGE CASE: different lengths (e.g. one engine stopped at EOS a token
    earlier due to a subtle numerical difference) -- this should return
    False cleanly, not raise on a shape mismatch.
    """
    raise NotImplementedError