"""
Base Engine interface.

Every inference technique (naive, kv_cache, paged, batched, speculative,
quantized) implements this same interface. This is what lets bench/runner.py
sweep across engines with identical calling code, and what lets tests compare
one engine's output against another's.

DESIGN NOTE (think about this before implementing):
    Should `generate()` be implemented once here in the base class, built out
    of `prefill()` + repeated `decode_step()` calls -- with subclasses only
    overriding the two primitives? Or should every engine implement its own
    `generate()`?

    Hint: continuous batching and speculative decoding don't cleanly fit a
    "prefill once, then decode one token at a time" loop -- batching admits
    new sequences mid-flight, and speculative decoding decodes K tokens per
    verify step, not 1. Decide now whether your abstraction can survive
    Week 3, or whether you're going to be fighting it by then.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import time


@dataclass
class GenerationConfig:
    """Sampling / decoding configuration shared across engines."""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = False  # False => greedy. Correctness tests rely on this
                              # being deterministic when False.
    eos_token_id: Optional[int] = None


@dataclass
class GenerationResult:
    """
    Standard return type for engine.generate(), so bench/runner.py and
    tests/ don't need to know which engine produced this.
    """
    input_ids: torch.Tensor          # (batch, prompt_len) original prompt
    output_ids: torch.Tensor         # (batch, prompt_len + generated_len)
    generated_ids: torch.Tensor      # (batch, generated_len) new tokens only
    # Per-step wall-clock timestamps, len == generated_len + 1 (prefill is index 0).
    # This is what latency.py will slice TTFT / TPOT out of -- don't compute
    # TTFT/TPOT here, just record raw timestamps. Keep this engine-agnostic.
    step_timestamps: list[float] = field(default_factory=list)
    # Anything engine-specific worth logging (e.g. spec decoding acceptance
    # rate, number of paged-attention blocks used). Optional, engine fills
    # in what's relevant.
    generate_start: float = 0.0
    extra: dict = field(default_factory=dict)


class Engine(ABC):
    """
    Common interface for all inference engines.

    Subclasses must implement `prefill` and `decode_step` at minimum.
    `generate` has a default implementation below that chains them --
    override it if your engine's control flow doesn't fit the
    prefill-then-loop-single-token-decode-steps pattern (see design note
    above; speculative and batched engines will very likely need to).
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Run the forward pass over the full prompt once.

        Must return whatever state your decode_step needs to continue
        generation (e.g. logits for the last position + any cache object).
        The exact return type is up to you -- just be consistent within
        your own engine, since generate() below only calls prefill/decode_step
        through you, not through inspecting internals.

        EDGE CASES to handle:
        - attention_mask is None (no padding, single sequence)
        - batch_size > 1 with left-padding (needed later for batching engine,
          but design prefill() now so it doesn't have to be rewritten)
        """
        raise NotImplementedError

    @abstractmethod
    def decode_step(self, state):
        """
        Given the state returned by prefill() (or the previous decode_step),
        produce the next token's logits and updated state.

        Returns: (logits_for_next_token, new_state)

        EDGE CASE: what happens when a sequence in the batch has already hit
        EOS but others haven't? (You don't have to solve this in Week 1's
        naive/kv_cache engines with batch_size=1, but decide now how you'll
        extend this signature later without breaking it.)
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, config: GenerationConfig,
                 attention_mask: Optional[torch.Tensor] = None) -> GenerationResult:
        """
        Default generate loop: prefill once, then decode_step in a loop
        until max_new_tokens or eos_token_id.

        1. Record a timestamp (time.perf_counter()) immediately after prefill
           returns, and after every decode_step -- append to step_timestamps.
        2. Apply sampling per `config` (greedy if not do_sample, else
           temperature/top_k/top_p) to select the next token from logits.
        3. Stop early if config.eos_token_id is produced (but still return
           a well-formed GenerationResult -- don't pad or crash).
        4. Must work correctly for batch_size == 1 first. Don't worry about
           multi-sequence batching here -- that's the batched engine's job
           in Week 3.

        THINK ABOUT: where exactly does timestamp[0] go relative to the
        prefill call -- before or after? This determines what "TTFT" ends up
        measuring, so get it right rather than fixing it after Week 2's
        engines have already copied the pattern.
        """
        
        generate_start_time = time.perf_counter()

        state = self.prefill(input_ids, attention_mask)

        step_timestamps = []

        generated_tokens = []
        for _ in range(config.max_new_tokens):
            # Call decode_step to get logits for the next token``
            logits, state = self.decode_step(state)

            # Pick a token from logits
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy decoding for now

            generated_tokens.append(next_token_id)
            step_timestamps.append(time.perf_counter())

            # Check for EOS token
            if config.eos_token_id is not None and next_token_id.item() == config.eos_token_id:
                break

            # state should be updated to include the new token for the next decode_step
            state = torch.cat([state, next_token_id], dim=-1)
        
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
            extra={}
        )

        

