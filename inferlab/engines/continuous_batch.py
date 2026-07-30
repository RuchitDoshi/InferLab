"""
Simplified continuous batching scheduler.

Admits waiting sequences into a fixed-size pool of "active" slots, prefills
new admissions, decodes one step for every active sequence, evicts
finished (EOS-hit) sequences and frees their PagedKVCache blocks for reuse
by the next admission.

SCOPE: sequences are processed via separate qwen_full_forward calls (looped
per sequence), not a single batched multi-sequence tensor op -- this
demonstrates PagedKVCache's shared-pool benefit under concurrent,
staggered sequence lifetimes, without requiring custom batched-attention
kernels (out of scope for a PyTorch-level reimplementation).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from inferlab.attention.from_scratch_layer import qwen_full_forward
from inferlab.attention.paged_kv_cache import PagedKVCache
from inferlab.engines.kv_cache import KVCache

class PrivateCacheScheduler:
    def __init__(self, model, max_active: int = 4):
        """
        TODO: same as ContinuousBatchScheduler.__init__, EXCEPT:
        - no single self.cache -- instead self.caches = {} (dict,
          sequence_id -> KVCache instance, populated lazily on admission)
        - everything else (self.max_active, self.waiting, self.active,
          self.finished, self._next_sequence_id) stays identical
        """
        self.model = model
        self.max_active = max_active
        self.waiting = []
        self.active = {}
        self.finished = {}
        self._next_sequence_id = 0
        self.caches = {}  # sequence_id -> KVCache instance, populated lazily on admission


    def submit(self, input_ids, max_new_tokens, eos_token_id):
        """TODO: identical to ContinuousBatchScheduler.submit() -- copy it."""
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self.waiting.append({
            "sequence_id": sequence_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        })
        return sequence_id

    def _admit_one(self):
        """
        TODO: identical to ContinuousBatchScheduler._admit_one(), EXCEPT:
        - create self.caches[sequence_id] = KVCache() right before prefill
        - call qwen_full_forward(input_ids, self.model, self.caches[sequence_id])
          -- NO sequence_id argument passed to qwen_full_forward (KVCache
          doesn't need one)
        """
        if self.waiting and len(self.active) < self.max_active:
            admission = self.waiting.pop(0)
            sequence_id = admission["sequence_id"]
            input_ids = admission["input_ids"]
            max_new_tokens = admission["max_new_tokens"]
            eos_token_id = admission["eos_token_id"]

            self.caches[sequence_id] = KVCache()
            hidden_states = qwen_full_forward(input_ids, self.model, self.caches[sequence_id])

            # Project to logits and sample the first token
            logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            state = SequenceState(
                input_ids=input_ids,
                sequence_id=sequence_id,
                most_recent_token=next_token,
                generated_tokens=[next_token],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                finished=False
            )
            self.active[sequence_id] = state

    def _decode_step(self):
        """
        TODO: identical to ContinuousBatchScheduler._decode_step(), EXCEPT
        each call to qwen_full_forward uses self.caches[state.sequence_id]
        as the cache, with no sequence_id argument.
        """
        for sequence_id, state in list(self.active.items()):
            if state.finished:
                continue

            input_ids = state.most_recent_token
            hidden_states = qwen_full_forward(input_ids, self.model, self.caches[state.sequence_id])

            # Project to logits and sample the first token
            logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            state.generated_tokens.append(next_token)
            state.most_recent_token = next_token

            # Check for EOS or max_new_tokens
            if (state.eos_token_id is not None and next_token.item() == state.eos_token_id) or \
               (len(state.generated_tokens) >= state.max_new_tokens):
                state.finished = True

    def _evict_finished(self):
        """
        TODO: same pattern (snapshot finished IDs first, then process),
        EXCEPT instead of calling cache.evict(sequence_id), just remove
        self.caches[sequence_id] entirely (e.g. self.caches.pop(sequence_id,
        None)) so its GPU memory can be freed. Still move the SequenceState
        from self.active to self.finished, same as before.
        """
        finished_ids = [seq_id for seq_id, state in self.active.items() if state.finished]
        for sequence_id in finished_ids:
            self.caches.pop(sequence_id, None)
            self.finished[sequence_id] = self.active.pop(sequence_id)

    @torch.no_grad()
    def run(self):
        """TODO: identical to ContinuousBatchScheduler.run() -- copy it."""
        while self.waiting or self.active:
            self._admit_one()
            self._decode_step()
            self._evict_finished()

        return self.finished




@dataclass
class SequenceState:
    """TODO: what does ONE active sequence need to track?
    - sequence_id
    - most recent token (what gets fed into the NEXT qwen_full_forward call)
    - generated_tokens so far (list, for building the final result)
    - max_new_tokens for this sequence (each sequence might want a
      different length -- or keep it simple: one shared config for now?)
    - eos_token_id
    - finished flag
    """
    input_ids: torch.Tensor
    sequence_id: int
    most_recent_token: torch.Tensor
    generated_tokens: list = field(default_factory=list)
    max_new_tokens: int = 50
    eos_token_id: int = None
    finished: bool = False

    def generated_ids(self) -> torch.Tensor:
        """
        TODO: torch.cat self.generated_tokens along the sequence dimension,
        same pattern used everywhere else today (torch.cat(tokens, dim=-1)).
        Handle the empty case gracefully (no tokens generated at all) --
        same pattern as every other engine's generate().
        """
        if not self.generated_tokens:
            return torch.empty(0, dtype=self.most_recent_token.dtype, device=self.most_recent_token.device)
        return torch.cat(self.generated_tokens, dim=-1)


class ContinuousBatchScheduler:
    def __init__(self, model, max_active: int = 4, block_size: int = 16, num_blocks: int = 300):
        """
        TODO:
        - self.model = model
        - build ONE shared PagedKVCache (derive num_layers/num_kv_heads/
          head_dim from model.config, same pattern as FromScratchEngine's
          __init__)
        - self.max_active -- cap on concurrently active sequences
        - self.waiting = []  (queue of not-yet-admitted prompts)
        - self.active = {}   (sequence_id -> SequenceState)
        - self.finished = {} (sequence_id -> SequenceState, for results)
        - self._next_sequence_id = 0  (simple incrementing counter)
        """
        self.model = model
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.cache = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=next(self.model.parameters()).dtype,
            device="cuda"
        )
        self.max_active = max_active
        self.waiting = []
        self.active = {}
        self.finished = {}
        self._next_sequence_id = 0

    def submit(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id):
        """
        TODO: add a new request to self.waiting. What does the waiting
        queue need to hold for a not-yet-admitted sequence -- just the
        prompt input_ids and its generation config? Return the assigned
        sequence_id so the caller can look up its result later.
        """
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self.waiting.append({
            "sequence_id": sequence_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        })
        return sequence_id

    def _admit_one(self):
        """
        TODO: if self.waiting is non-empty AND len(self.active) < self.max_active:
        pop one waiting request, assign it a sequence_id, run prefill via
        qwen_full_forward(prompt_input_ids, self.model, self.cache,
        sequence_id=new_id), project to logits, sample first token,
        create a SequenceState, add to self.active.
        """

        if self.waiting and len(self.active) < self.max_active:
            request = self.waiting.pop(0)
            sequence_id = request["sequence_id"]
            input_ids = request["input_ids"]
            max_new_tokens = request["max_new_tokens"]
            eos_token_id = request["eos_token_id"]

            # Prefill the cache with the prompt
            hidden_states = qwen_full_forward(input_ids, self.model, self.cache, sequence_id=sequence_id)

            # Project to logits and sample the first token
            logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            # Create a SequenceState and add it to active
            state = SequenceState(
                input_ids=input_ids,
                sequence_id=sequence_id,
                most_recent_token=next_token,
                generated_tokens=[next_token],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                finished=False
            )
            self.active[sequence_id] = state


    def _decode_step(self):
        """
        TODO: for EACH sequence_id in self.active (loop, not batched --
        see design discussion), call qwen_full_forward(most_recent_token,
        self.model, self.cache, sequence_id=sequence_id), project to
        logits, sample next token, append to that sequence's
        generated_tokens, update its most-recent-token.
        """
        for sequence_id, state in self.active.items():
            if state.finished:
                continue  # Skip finished sequences

            # Call qwen_full_forward for the most recent token
            hidden_states = qwen_full_forward(state.most_recent_token, self.model, self.cache, sequence_id=sequence_id)

            # Project to logits and sample the next token
            logits = F.linear(hidden_states[:, -1:, :], self.model.lm_head.weight)
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # Greedy decoding for now

            # Append to generated tokens and update most recent token
            state.generated_tokens.append(next_token)
            state.most_recent_token = next_token

            # Check if the sequence has finished
            if next_token.item() == state.eos_token_id or len(state.generated_tokens) >= state.max_new_tokens:
                state.finished = True


    def _evict_finished(self):
        """
        TODO: for each active sequence, check if it just hit EOS or its
        max_new_tokens -- if so, call self.cache.evict(sequence_id) to
        free its blocks, move it from self.active to self.finished.
        """
        finished_sequence_ids = [seq_id for seq_id, state in self.active.items() if state.finished]

        for sequence_id in finished_sequence_ids:
            # Evict the sequence from the cache
            self.cache.evict(sequence_id)

            # Move the sequence from active to finished
            self.finished[sequence_id] = self.active.pop(sequence_id)

    @torch.no_grad()
    def run(self):
        """
        TODO: the main loop -- while self.waiting or self.active:
            self._admit_one()
            if self.active: self._decode_step()
            self._evict_finished()
        Return self.finished (or just let callers inspect it after run()).
        """
        while self.waiting or self.active:
            self._admit_one()
            if self.active:
                self._decode_step()
            self._evict_finished()

        return self.finished

