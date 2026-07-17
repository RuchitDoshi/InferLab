
import torch
import torch.nn.functional as F
import time
from typing import Optional
from inferlab.attention.from_scratch_layer import qwen_full_forward
from inferlab.engines.kv_cache import KVCache
from inferlab.engines.base import Engine, GenerationConfig, GenerationResult

class FromScratchEngine(Engine):
    """
    Engine wrapping the from-scratch qwen_full_forward (embedding -> 24
    hand-written decoder layers -> final norm -> lm_head projection),
    with hand-written GQA-aware KV caching (engines/kv_cache.py's
    multi-layer KVCache).
    """
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Run the forward pass over the full prompt once, populating a
        KVCache() instance for all layers. Return the last position's
        logits and the KVCache instance.
        """
        pass

    def decode_step(self, state):
        """
        Given the state returned by prefill() (or the previous decode_step),
        produce the next token's logits and updated state.

        Returns: (logits_for_next_token, new_state)
        """
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, config: GenerationConfig,
                 attention_mask: Optional[torch.Tensor] = None) -> GenerationResult:
        """
        TODO:
        1. generate_start_time = time.perf_counter()
        2. Create ONE KVCache() instance, local to this call.
        3. "Prefill": call qwen_full_forward(input_ids, self.model, kv_cache)
           -> hidden_states. Project last position to logits via
           F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight).
           Sample first token (argmax). Append to generated_tokens, record
           timestamp.
        4. Loop up to max_new_tokens - 1 more times (same pattern as
           KVCacheEngine.generate()):
           - check EOS on the last-generated token BEFORE continuing
           - call qwen_full_forward(next_token, self.model, kv_cache) --
             next_token shape (batch, 1), the SAME kv_cache object (still
             mutating in place)
           - project to logits, sample, append, timestamp
        5. Build and return GenerationResult, same shape as every other
           engine today (input_ids, output_ids, generated_ids,
           step_timestamps, generate_start).
        """
        generate_start_time = time.perf_counter()
        kv_cache = KVCache()

        hidden_states = qwen_full_forward(input_ids, self.model, kv_cache=kv_cache)
        logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
        next_token_id = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

        timestamp_after_prefill = time.perf_counter()
        generated_tokens = [next_token_id]  # Start with the first generated token
        step_timestamps = [timestamp_after_prefill]

        for _ in range(max(config.max_new_tokens - 1, 0)):  # Already generated one token
            # Check for EOS token
            if config.eos_token_id is not None and next_token_id.item() == config.eos_token_id:
                break
            
            # Call qwen_full_forward for the next token
            hidden_states = qwen_full_forward(next_token_id, self.model, kv_cache=kv_cache)
            logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
            next_token_id = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            generated_tokens.append(next_token_id)
            step_timestamps.append(time.perf_counter())
        
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
            generate_start=generate_start_time
        )
