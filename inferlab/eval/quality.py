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

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

def compute_perplexity_windowed(model, tokenizer, text: str, device: str = "cuda",
                                  window_size: int = 512, stride: int = 256) -> float:
    """
    Sliding-window perplexity for long text, avoiding the O(seq_len) logits
    memory blowup of scoring the whole sequence in one forward pass.
    Tokenizes ONCE, slides over TOKEN positions (not characters), and
    masks the overlapping "context" portion of each window (-100 sentinel,
    HF's convention for "ignore this position in the loss") so tokens
    never get double-counted across overlapping windows.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    total_len = input_ids.shape[1]

    if total_len <= window_size:
        return compute_perplexity(model, tokenizer, text, device=device)

    total_loss = 0.0
    total_tokens = 0
    prev_end = 0

    for start in range(0, total_len, stride):
        end = min(start + window_size, total_len)
        window_ids = input_ids[:, start:end]

        labels = window_ids.clone()
        context_len = prev_end - start  # how much of this window overlaps the previous one
        if context_len > 0:
            labels[:, :context_len] = -100  # don't re-score already-scored tokens

        with torch.no_grad():
            outputs = model(input_ids=window_ids, labels=labels)

        num_scored = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_scored
        total_tokens += num_scored
        prev_end = end

        if end == total_len:
            break

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()
    


def exact_match(reference_ids: torch.Tensor, candidate_ids: torch.Tensor) -> bool:
    """
    True if candidate_ids equals reference_ids exactly
    (same length, same tokens). Used to compare an engine's greedy output
    against NaiveEngine's greedy output on the same prompt.

    EDGE CASE: different lengths (e.g. one engine stopped at EOS a token
    earlier due to a subtle numerical difference) -- this should return
    False cleanly, not raise on a shape mismatch.
    """
    if reference_ids.shape != candidate_ids.shape:
        return False
    return torch.equal(reference_ids, candidate_ids)