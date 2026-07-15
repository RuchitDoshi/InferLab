"""
KV-cache engine: standard incremental decoding using a hand-managed KV cache.

IMPORTANT DESIGN DECISION: you have two honest options here, and you should
pick deliberately rather than by default:

  (a) Use HF's built-in `past_key_values` mechanism (pass
      use_cache=True, thread past_key_values through each call). This is
      the "HF backbone" approach -- correct, fast to build, but you're
      relying on HF's cache management, not writing your own.

  (b) Write your own KVCache class (tensor allocation, per-layer K/V
      storage, an `update()` method that appends new K/V and returns the
      full cache for attention to read) and monkeypatch/replace the
      attention module's forward to read/write YOUR cache object instead
      of HF's.

Given your stated goal ("implement from scratch AND benchmark rigorously"),
you should do (b) -- but (a) is a legitimate fallback if Week 1 is taking
too long and you'd rather bank a working baseline first, then retrofit (b)
once the metrics harness exists. Decide now; don't drift into (a) by
accident just because it's easier to get `generate()` working quickly.

This file assumes you're doing (b). If you pick (a), the KVCache class
below can be a thin wrapper around past_key_values instead -- keep the
same interface either way so decode_step()'s signature doesn't change
later.
"""

from dataclasses import dataclass, field
from logging import config
import time
from typing import Optional

import torch

from .base import Engine, GenerationConfig, GenerationResult


@dataclass
class KVCache:
    """
    Own your own KV cache. One entry per transformer layer.

    TODO(ruchit): decide the tensor shape convention up front and write it
    here as a comment, e.g. (batch, n_kv_heads, seq_len, head_dim). This
    shape decision matters a lot in Week 2 -- PagedAttention will replace
    this contiguous-tensor cache with a block-table version, so know exactly
    what contract you're breaking/preserving when you get there.
    """
    # keys[layer_idx] -> tensor
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        TODO(ruchit): append new_k/new_v (for the newly computed token(s))
        to this layer's cache along the sequence dimension, and return the
        FULL k, v tensors for that layer (existing + new) for attention to
        use.

        EDGE CASE: first call for a given layer_idx -- self.keys may not
        have an entry yet. Handle initialization vs append distinctly
        rather than assuming keys[layer_idx] already exists.
        """
        raise NotImplementedError

    def current_length(self) -> int:
        """TODO(ruchit): return current cached sequence length (0 if empty)."""
        raise NotImplementedError

    def memory_bytes(self) -> int:
        """
        TODO(ruchit): compute actual bytes currently held by this cache
        (sum tensor.numel() * element_size() across keys+values). This is
        your "measured" number to validate against the theoretical formula
        in metrics/memory.py during Week 2's GQA analysis. Write it now
        while the cache is fresh in your mind, even though you won't use
        it until Week 2 -- it's a two-line method and easy to forget later.
        """
        raise NotImplementedError


class KVCacheEngine(Engine):
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        TODO(ruchit): run forward pass ONCE on the full prompt, populate a
        fresh KVCache from it, return (logits_last_position, cache).

        THINK ABOUT: if you're replacing attention's forward function to
        write into your own KVCache (option b above), where does that
        monkeypatch/injection happen -- once per Engine instance, or per
        prefill() call? Get this right now; speculative decoding in Week 3
        will need two live caches at once (draft + target), so your
        patching mechanism needs to not leak state between them.
        """
        outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        return logits, past_key_values


    def decode_step(self, state):
        """
        state = (cache,) or similar -- whatever prefill returns minus the
        logits.

        TODO(ruchit): forward pass on ONLY the newly generated token
        (shape (batch, 1)), using the existing cache for all prior context.
        Update the cache, return (logits_next_token, updated_state).

        CORRECTNESS CHECK (do this yourself before writing the pytest):
        run this engine and NaiveEngine on the identical prompt with
        do_sample=False, and diff the generated token ids. If they don't
        match exactly, the bug is almost always one of: (1) off-by-one on
        which position's logits you're sampling from, (2) attention_mask
        not accounting for cached length, (3) position_ids not offset
        correctly when only feeding 1 new token to a model that expects
        them to reflect the full sequence position.
        """
        next_input = state.get("next_input")  # This should be the last generated token
        past_key_values = state.get("past_key_values")  # This should be the cached key/values

        outputs = self.model(next_input, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        new_past_key_values = outputs.past_key_values
        return logits, new_past_key_values


    
    def generate(self, input_ids: torch.Tensor, config: GenerationConfig,
                 attention_mask: Optional[torch.Tensor] = None) -> GenerationResult:
        """
        TODO(ruchit): implement this. Requirements:
        1. prefill once, then decode_step in a loop until max_new_tokens or
           eos_token_id.
        2. record timestamps after prefill and after every decode_step.
        3. apply sampling per config (greedy if not do_sample, else
           temperature/top_k/top_p).
        4. stop early if eos_token_id is produced (but still return a
           well-formed GenerationResult -- don't pad or crash).
        5. must work correctly for batch_size == 1 first. Don't worry about
           multi-sequence batching here -- that's the batched engine's job
           in Week 3.
        """ 
        generate_start_time = time.perf_counter()

        #call prefill to get the initial state (logits and cache)
        logits, past_key_values = self.prefill(input_ids, attention_mask)
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy decoding for now
        
        timestamp_after_prefill = time.perf_counter()
        generated_tokens = [next_token_id]  # Start with the first generated token

        state = {
            "next_input": next_token_id,
            "past_key_values": past_key_values
        }

        # Timestamps for each step, starting with the timestamp after prefill
        step_timestamps = [timestamp_after_prefill]

        for _ in range(max(config.max_new_tokens - 1, 0)):  # Already generated one token
            # Check for EOS token
            if config.eos_token_id is not None and next_token_id.item() == config.eos_token_id:
                break
            
            # Call decode_step to get logits for the next token
            logits, past_key_values = self.decode_step(state)

            # Pick a token from logits
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy decoding for now

            generated_tokens.append(next_token_id)
            step_timestamps.append(time.perf_counter())


            # Update the state with the new token for the next decode_step
            state = {
                "next_input": next_token_id,
                "past_key_values": past_key_values
            }
        
        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
        else:
            generated_ids = torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype)
        output_ids = torch.cat([input_ids, generated_ids], dim=-1)

        return GenerationResult(
            input_ids=input_ids,
            output_ids=output_ids,
            generated_ids=generated_ids,
            step_timestamps=step_timestamps,
            generate_start=generate_start_time,
        )
