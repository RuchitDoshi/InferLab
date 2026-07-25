"""
Simplified PagedAttention: block-based KV cache allocation instead of
KVCache's contiguous torch.cat growth. Motivation: torch.cat reallocates
and copies the ENTIRE cache history on every single token, which is fine
at small single-sequence scale but becomes a real problem with many
concurrent, differently-growing sequences sharing GPU memory (fragmentation
-- plenty of total free memory, but no single contiguous chunk big enough).
Paged allocation uses fixed-size blocks from a shared pool, addressed
indirectly via a per-sequence block table -- growing a sequence means
grabbing one more block, never copying existing data.

SCOPE: single sequence for now (no multi-sequence prefix sharing yet --
that's a real vLLM feature but adds complexity beyond this first pass).
Known, honest tradeoff: internal fragmentation up to (block_size - 1)
wasted token-slots in the last, partially-filled block of a sequence.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class PagedKVCache:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16
    num_blocks: int = 100  # total blocks in the pool, across all layers
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    def __post_init__(self):
        """
        TODO: allocate the actual block pools -- one tensor per layer,
        each shaped (num_blocks, block_size, num_kv_heads, head_dim), for
        BOTH keys and values (so two lists of tensors, or one combined
        structure -- your choice, be consistent).

        Also initialize:
        - free_blocks: a list/set of all block indices [0, 1, ..., num_blocks-1],
          representing every block that's currently unused and available.
        - block_table: this sequence's ordered list of allocated block
          indices, starts empty.
        - a way to track current_length() (total real tokens cached so far
          for this sequence) -- can you derive this from len(block_table)
          and how many tokens are in the LAST block, or is it simpler to
          just track it as its own counter, incremented by 1 each update?
        """
        self.key_pools = [
            torch.zeros((self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
                        dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_pools = [
            torch.zeros((self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
                        dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.free_blocks = [list(range(self.num_blocks)) for _ in range(self.num_layers)]
        self.block_table = [[] for _ in range(self.num_layers)]
        self._current_length = [0 for _ in range(self.num_layers)]  # Track current length per layer


    def allocate_block(self, layer_idx: int) -> int:
        """
        TODO: pop and return one index from free_blocks. What should
        happen if free_blocks is empty (pool exhausted)? This is a real
        failure mode worth handling explicitly (raise a clear error)
        rather than letting it crash confusingly later.
        """
        if not self.free_blocks[layer_idx]:
            raise RuntimeError(f"No free blocks available (pool size: {self.num_blocks}).")
        return self.free_blocks[layer_idx].pop()

    def _update_single_token(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Update the key and value caches for a single token (seq_len=1) for the given layer.

        new_k, new_v shape (batch, num_kv_heads, 1, head_dim).

        Returns the full reconstructed key and value caches for this layer
        after the update.

        This method handles the block allocation and writing in place, as well
        as reconstructing the full k, v tensors for the layer after the update.
        """
        block_table_position = self._current_length[layer_idx] // self.block_size
        slot_within_block = self._current_length[layer_idx] % self.block_size

        if block_table_position >= len(self.block_table[layer_idx]):
            new_block_idx = self.allocate_block(layer_idx)
            self.block_table[layer_idx].append(new_block_idx)

        actual_block_index = self.block_table[layer_idx][block_table_position]

        # Write in place
        self.key_pools[layer_idx][actual_block_index, :,slot_within_block] = new_k.squeeze(2).squeeze(0)
        self.value_pools[layer_idx][actual_block_index,:,slot_within_block] = new_v.squeeze(2).squeeze(0)

        # Increment length counter
        self._current_length[layer_idx] += 1

        # Reconstruct full k, v for this layer
        full_k = torch.cat([self.key_pools[layer_idx][idx] for idx in self.block_table[layer_idx]], dim=1)[:, :self._current_length[layer_idx]].unsqueeze(0)  # Add the batch dimension back
        full_v = torch.cat([self.value_pools[layer_idx][idx] for idx in self.block_table[layer_idx]], dim=1)[:, :self._current_length[layer_idx]].unsqueeze(0)  # Add the batch dimension back

        return full_k, full_v

    def _update_batch(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Update the key and value caches for the given layer with new_k and new_v.

        new_k, new_v shape (batch, num_kv_heads, seq_len, head_dim).

        This method assumes that seq_len == 1 for simplicity. If seq_len > 1,
        it will raise an error. For batch updates with seq_len > 1, you can
        loop over the batch dimension and call update() for each item in the batch.

        Returns the full reconstructed key and value caches for this layer
        after the update.
        """
        seq_len = new_k.size(2)
        num_blocks_needed = torch.ceil(torch.tensor(self._current_length[layer_idx] + seq_len) / self.block_size).int().item()

        while len(self.block_table[layer_idx]) < num_blocks_needed:
            new_block_idx = self.allocate_block(layer_idx)
            self.block_table[layer_idx].append(new_block_idx)

        # Write in place for each token in the new_k/new_v
        tokens_written = 0
        while tokens_written < seq_len:
            slot_start = (self._current_length[layer_idx] + tokens_written) % self.block_size
            chunk_size = min(self.block_size - slot_start, seq_len - tokens_written)
            block_table_position = (self._current_length[layer_idx] + tokens_written) // self.block_size

            self.key_pools[layer_idx][self.block_table[layer_idx][block_table_position], :, slot_start:slot_start + chunk_size] = new_k[:, :, tokens_written:tokens_written + chunk_size, :].squeeze(0)
            self.value_pools[layer_idx][self.block_table[layer_idx][block_table_position], :, slot_start:slot_start + chunk_size] = new_v[:, :, tokens_written:tokens_written + chunk_size, :].squeeze(0)
            tokens_written += chunk_size

        self._current_length[layer_idx] += seq_len

        full_k = torch.cat([self.key_pools[layer_idx][idx] for idx in self.block_table[layer_idx]], dim=1)[:, :self._current_length[layer_idx]].unsqueeze(0)  # Add the batch dimension back
        full_v = torch.cat([self.value_pools[layer_idx][idx] for idx in self.block_table[layer_idx]], dim=1)[:, :self._current_length[layer_idx]].unsqueeze(0)  # Add the batch dimension back


        return full_k, full_v

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Update the key and value caches for the given layer with new_k and new_v.

        new_k, new_v shape (batch, num_kv_heads, seq_len, head_dim).

        If seq_len > 1, this will loop over each position and call
        _update_single_token for each token.

        Returns the full reconstructed key and value caches for this layer
        after the update.
        """
        seq_len = new_k.size(2)

        if seq_len > 1:
            full_k, full_v = self._update_batch(layer_idx, new_k, new_v)
        else:
            # If seq_len == 1, we can directly call _update_single_token without looping
            full_k, full_v = self._update_single_token(layer_idx, new_k, new_v)

        return full_k, full_v


    def current_length(self, layer_idx: int) -> int:
        """TODO: however you decided to track this in __post_init__."""
        return self._current_length[layer_idx]