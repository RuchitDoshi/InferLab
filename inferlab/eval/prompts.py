"""
Fixed benchmark prompt set: short / medium / long context tiers.

Long-context tier exists specifically to stress-test KV cache vs naive
(Week 1) and contiguous vs paged cache (Week 2) -- these differences are
dramatic at long context and boring at short context, so don't skip
building this out just because short prompts are easier to hand-write.

TODO(ruchit): decide your source for the long-context tier. Recommend
pulling real text (not lorem-ipsum-style filler) from a reproducible public
source so the benchmark is citable/defensible in a README, e.g.:
  - `datasets` library: load a slice of wikitext-103 or a subset of The Pile
  - concatenate/truncate to your target token counts (~128 / ~1024 / ~4096)
Keep the actual prompt text out of this file if it's large -- load it from
a small local .jsonl / cache it via `datasets`, and have this module expose
just the loading function + tier config.
"""

from dataclasses import dataclass


@dataclass
class PromptTier:
    name: str            # "short" | "medium" | "long"
    target_tokens: int    # approx token count to truncate/pad to
    prompts: list[str]    # the actual prompt strings


def load_benchmark_prompts(tokenizer) -> dict[str, PromptTier]:
    """
    TODO(ruchit): build and return {"short": PromptTier(...), "medium": ...,
    "long": ...}. Suggested target_tokens: short=128, medium=1024, long=4096.

    Use `tokenizer` to verify/truncate each prompt to its tier's target
    length (encode, truncate to target_tokens, decode back to text OR just
    keep token ids directly and skip decode -- decide which representation
    downstream code wants; probably token ids, since bench/runner.py will
    feed these straight into engine.generate() without re-tokenizing).

    Aim for 3-5 prompts per tier minimum, so latency numbers aren't just
    single-sample noise.
    """
    raise NotImplementedError