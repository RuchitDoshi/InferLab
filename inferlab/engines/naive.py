"""
Naive engine: no KV cache. At every decode step, re-run the full forward
pass over the *entire* sequence generated so far.

This is intentionally the slow baseline -- its only job is to be correct
and simple, so every other engine can be tested against it. Every other
engine's correctness test should assert: same greedy output as NaiveEngine,
given the same prompt and model.

Why this matters for the metrics story: this engine's decode-step cost grows
with sequence length (recomputing attention over a growing sequence each
step) -- roughly O(n) per step, O(n^2) total for n tokens, since nothing is
cached. Your Week 1 deliverable plot (naive vs kv_cache latency crossover)
depends on this engine being a faithful, unoptimized baseline -- resist the
urge to sneak in any caching "for speed."
"""

from typing import Optional

import torch

from .base import Engine


class NaiveEngine(Engine):
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        TODO(ruchit): Run the full forward pass on input_ids, no cache.
        Return whatever decode_step needs -- at minimum, the running
        sequence of token ids (since decode_step will re-run everything).

        Hint: since this engine never caches, `state` can just be the
        input_ids tensor itself (decode_step appends to it and re-forwards).
        Keep it that simple -- don't build cache machinery here, that's
        kv_cache.py's job.
        """
        return input_ids

    def decode_step(self, state):
        """
        TODO(ruchit): state is the full sequence so far. Re-run the model
        on the WHOLE sequence (model(state) -- no past_key_values), take
        logits at the last position, return (logits, updated_state) where
        updated_state hasn't appended the new token yet -- generate() does
        that after sampling. Decide whether decode_step should append
        internally or leave that to generate(); pick one and be consistent,
        since kv_cache.py's decode_step signature should match this one.

        EDGE CASE: make sure you're indexing logits at [:, -1, :] (last
        position across the full sequence), not [:, 0, :] or the whole
        tensor -- an easy off-by-one here silently produces garbage tokens
        that still "look like" output, so don't rely on eyeballing.
        """
        logits = self.model(state).logits
        return logits[:, -1, :], state
        