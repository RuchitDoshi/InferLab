"""
Speculative decoding: a small draft model proposes K tokens sequentially,
a larger target model verifies all K in one batched forward pass, accepts
the longest matching prefix, and samples a replacement (plus possibly one
bonus token) from the target's own logits at the point of divergence.

SCOPE: draft model's cache is rebuilt from scratch each round (re-prefill
over the full accepted-so-far sequence) rather than incrementally rolled
back on rejection -- avoids needing shrink/rollback support in
KVCache/PagedKVCache, at the cost of O(n^2) draft-side re-prefill over
many rounds. Accepted as reasonable for a first version, same as Week 1's
naive engine and PagedAttention's first prefill pass -- revisit only if
benchmarking shows it's an actual bottleneck.
"""

import time
import torch
import torch.nn.functional as F

from inferlab.attention.from_scratch_layer import qwen_full_forward
from inferlab.engines.kv_cache import KVCache
from inferlab.engines.base import Engine, GenerationResult


class SpeculativeEngine(Engine):
    def __init__(self, draft_model, target_model, device: str = "cuda", num_draft_tokens: int = 4):
        """
        TODO: super().__init__(target_model, device) -- self.model becomes
        the target (matches base class convention: self.model is "the"
        model this engine represents for benchmarking purposes).
        Store self.draft_model separately. Store self.num_draft_tokens (K).
        """
        super().__init__(target_model, device)
        self.draft_model = draft_model
        self.num_draft_tokens = num_draft_tokens

    def prefill(self, input_ids, attention_mask = None):
        pass

    def decode_step(self, state):
        """
        Given the state returned by prefill() (or the previous decode_step),
        produce the next token's logits and updated state.

        Returns: (logits_for_next_token, new_state)
        """
        pass
    
    def _get_logits(self, model, hidden_states):
        """
        TODO: small shared helper -- F.linear(hidden_states, weight) using
        the TARGET model's lm_head (used for verification logits). Given
        tied embeddings, does the draft model need its OWN lm_head
        projection too (yes -- it's a different model, different weights)?
        Consider whether this helper needs a `model` parameter to pick
        which model's lm_head to use, given both draft and target need
        logits at different points.
        """
        return F.linear(hidden_states, model.lm_head.weight)

    @torch.no_grad()
    def generate(self, input_ids, config, attention_mask=None):
        """
        TODO: the main loop. Sketch (fill in incrementally):

        1. generate_start_time = time.perf_counter()
        2. target_cache = KVCache()
        3. Prefill TARGET on the full prompt -> hidden_states -> logits at
           last position -> sample first token via argmax (exactly like
           FromScratchEngine's prefill step). Append to generated_tokens,
           timestamp.
        4. Track total_accepted = 0, total_drafted = 0 (for acceptance
           rate reporting later).
        5. Loop until EOS or max_new_tokens:
           a. Rebuild draft_cache = KVCache() fresh this round.
              Re-prefill DRAFT over [prompt + all accepted tokens so far]
              to populate draft_cache (the accepted "recompute" tradeoff).
           b. Draft K new tokens sequentially from draft_cache (ordinary
              decode steps on the draft model).
           c. Verify: ONE target forward pass over the K drafted tokens
              (using target_cache, which already holds the prompt +
              accepted-so-far -- NOT re-prefilled, unlike the draft).
              Get logits at K positions.
           d. Walk the K drafted tokens in order, compare each against
              the shift-by-one-aligned target logits (drafted token i
              checked against logits produced BEFORE token i was fed in).
              Accept while matching; stop at first mismatch or after all K.
           e. Determine actual next committed token(s): accepted prefix +
              either (mismatch: sample from target's logits AT the
              mismatch position) or (all K matched: bonus token from
              target's logits AFTER the last drafted token).
           f. Append committed tokens to generated_tokens (one timestamp
              per committed token, matching the project's established
              step_timestamps convention). Update total_accepted/
              total_drafted.
           g. Check EOS, check max_new_tokens, break if done.
        6. Build GenerationResult as usual. Consider: does GenerationResult
           need a new field for acceptance rate, or does that belong in
           `extra` (already exists on GenerationResult for exactly this
           kind of engine-specific metric)?
        """

        generate_start_time = time.perf_counter()
        target_cache = KVCache()
        hidden_states = qwen_full_forward(input_ids, self.model, kv_cache=target_cache)
        logits = self._get_logits(self.model, hidden_states[:, -1:, :])
        last_logits = logits
        next_token_id = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

        timestamp_after_prefill = time.perf_counter()
        generated_tokens = [next_token_id]  # Start with the first generated token
        step_timestamps = [timestamp_after_prefill]

        hidden_states = qwen_full_forward(next_token_id, self.model, kv_cache=target_cache)
        last_logits = self._get_logits(self.model, hidden_states)

        total_accepted = 0
        total_drafted = 0

        for _ in range(max(config.max_new_tokens - 1, 0)):  # Already generated one token
            if config.eos_token_id is not None and next_token_id.item() == config.eos_token_id:
                break

            if len(generated_tokens) >= config.max_new_tokens:
                break

            accepted_so_far = torch.cat([input_ids] + generated_tokens, dim=-1) if generated_tokens else input_ids
            
            drafted_tokens = []

            draft_cache = KVCache()  # Rebuild draft cache fresh this round
            hidden_states = qwen_full_forward(accepted_so_far, self.draft_model, kv_cache=draft_cache)
            draft_logits = self._get_logits(self.draft_model, hidden_states[:, -1:, :])
            next_token_id = torch.argmax(draft_logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            drafted_tokens.append(next_token_id)
            
            for i in range(self.num_draft_tokens - 1):
                #decode step for draft model 
                hidden_states = qwen_full_forward(next_token_id, self.draft_model, kv_cache=draft_cache)
                draft_logits = self._get_logits(self.draft_model, hidden_states[:, -1:, :])
                next_token_id = torch.argmax(draft_logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

                drafted_tokens.append(next_token_id)

            prev_target_cache_length = target_cache.current_length(0)
            drafted_tokens_tensor = torch.cat(drafted_tokens, dim=-1)
            hidden_states = qwen_full_forward(drafted_tokens_tensor, self.model, kv_cache=target_cache)
            target_logits = self._get_logits(self.model, hidden_states)

            rejected = False
            for i in range(self.num_draft_tokens):
                if len(generated_tokens) >= config.max_new_tokens:
                    break

                drafted_token_id = drafted_tokens_tensor[:,i:i+1]

                if i < 1:
                    drafted_token_id = drafted_tokens_tensor[:,i:i+1]
                    target_logit = torch.argmax(last_logits, dim=-1, keepdim=True).squeeze(1)
            
                else:
                    target_logit = torch.argmax(target_logits[:, i-1:i, :], dim=-1, keepdim=True).squeeze(1)

                if target_logit.item() == drafted_token_id.item():
                    total_accepted += 1
                    generated_tokens.append(drafted_token_id)
                    step_timestamps.append(time.perf_counter())
                else:
                    generated_tokens.append(target_logit)
                    step_timestamps.append(time.perf_counter())
                    rejected = True
                    break

            if rejected:
                target_cache.truncate(prev_target_cache_length + i)
                hidden_states = qwen_full_forward(target_logit, self.model, kv_cache=target_cache)
                target_logits = self._get_logits(self.model, hidden_states)
                next_token_id = target_logits
                last_logits = target_logits

            else:
                last_logits = target_logits[:,self.num_draft_tokens -1:, :]


            total_drafted += self.num_draft_tokens
            next_token_id = generated_tokens[-1]

        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
        else:
            generated_ids = torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype)
        output_ids = torch.cat([input_ids, generated_ids], dim=-1)

        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else None

        return GenerationResult(
            input_ids=input_ids,
            output_ids=output_ids,
            generated_ids=generated_ids,
            step_timestamps=step_timestamps,
            generate_start=generate_start_time,
            extra={"acceptance_rate": acceptance_rate, "total_accepted": total_accepted, "total_drafted": total_drafted},
        )
