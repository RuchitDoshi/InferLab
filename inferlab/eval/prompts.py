"""
Fixed benchmark prompt set: short / medium / long context tiers.

Long-context tier exists specifically to stress-test KV cache vs naive
(Week 1) and contiguous vs paged cache (Week 2) -- these differences are
dramatic at long context and boring at short context.

Source: wikitext-103-raw-v1 (Salesforce/wikitext on the HF Hub), streamed
rather than fully downloaded. Individual wikitext lines are often too
short to hit medium/long targets on their own, so lines are concatenated
until each prompt reaches its tier's target_tokens, then truncated exactly
to that length -- every prompt in a tier is the same token length, which
is what makes the tier-vs-tier latency comparison meaningful rather than
noisy.

Design decision: prompts are stored as already-tokenized input_ids
(torch.Tensor, shape (1, target_tokens)), not raw text. This means
tokenization happens once here, not on every bench/runner.py sweep --
tokenizer settings can't silently drift between runs, and runner.py can
feed these straight into engine.generate() with no extra step.
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset

MIN_LINE_TOKENS = 20  # skip near-empty lines (blank lines, bare headers)
                        # that would otherwise pollute concatenated prompts
                        # with junk


@dataclass
class PromptTier:
    name: str
    target_tokens: int
    prompts: list  # list[torch.Tensor], each shape (1, target_tokens)


def _build_tier_prompts(tokenizer, line_iterator, target_tokens: int, num_prompts: int):
    """
    Pull lines from line_iterator, concatenating (with a space) until the
    running token count reaches target_tokens, then truncate to exactly
    target_tokens. Repeat until num_prompts prompts are built.
    """
    prompts = []
    buffer_ids = []

    for line in line_iterator:
        line = line.strip()
        if not line:
            continue

        line_ids = tokenizer.encode(line, add_special_tokens=False)
        if len(line_ids) < MIN_LINE_TOKENS:
            continue

        buffer_ids.extend(line_ids)

        while len(buffer_ids) >= target_tokens:
            prompt_ids = buffer_ids[:target_tokens]
            prompts.append(torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0))
            buffer_ids = buffer_ids[target_tokens:]  # keep leftover for next prompt

            if len(prompts) >= num_prompts:
                return prompts

    return prompts  # may be short of num_prompts if the stream ran dry


def load_benchmark_prompts(tokenizer, num_prompts_per_tier: int = 5) -> dict:
    """
    Build and return {"short": PromptTier(...), "medium": ..., "long": ...}.
    target_tokens: short=128, medium=1024, long=4096.
    """
    tier_configs = [
        ("short", 128),
        ("medium", 1024),
        ("long", 4096),
    ]

    tiers = {}
    for name, target_tokens in tier_configs:
        # Fresh streaming iterator per tier -- simplest correct approach.
        stream = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        )
        line_iterator = (row["text"] for row in stream)

        prompts = _build_tier_prompts(tokenizer, line_iterator, target_tokens, num_prompts_per_tier)

        if len(prompts) < num_prompts_per_tier:
            raise RuntimeError(
                f"Only found {len(prompts)}/{num_prompts_per_tier} prompts for "
                f"tier '{name}' (target_tokens={target_tokens}) before the "
                f"stream ran dry. Try a larger dataset split or lower "
                f"num_prompts_per_tier."
            )

        tiers[name] = PromptTier(name=name, target_tokens=target_tokens, prompts=prompts)

    return tiers