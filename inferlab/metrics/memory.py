"""
Memory metrics: measured peak VRAM + theoretical KV cache size calculator.

The theoretical calculator is used properly in Week 2 (GQA analysis), but
stub it now since it's pure math with no engine dependency.
"""

import torch


def peak_vram_bytes(device: str = "cuda") -> int:
    """
    use torch.cuda.max_memory_allocated(device) to get peak
    allocated bytes since the last reset. Caller is responsible for calling
    torch.cuda.reset_peak_memory_stats(device) before the region they want
    to measure -- document that requirement clearly in the docstring so
    bench/runner.py uses it correctly (reset -> run -> read peak).
    """
    return torch.cuda.max_memory_allocated(device)


def theoretical_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # fp16/bf16 = 2 bytes
    batch_size: int = 1,
) -> int:
    """
    implement the standard KV cache memory formula:

        2 (K and V) * num_layers * num_kv_heads * head_dim * seq_len
        * dtype_bytes * batch_size

    This is the number you'll compare against KVCache.memory_bytes()
    (measured) in Week 2 to validate your cache implementation, and against
    a hypothetical num_kv_heads == num_attention_heads (MHA) config to
    quantify GQA's savings. Get the formula right now with a unit test
    against a hand-computed example -- don't discover an off-by-factor-of-2
    bug in Week 2 when it's tangled up with the GQA writeup.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes * batch_size