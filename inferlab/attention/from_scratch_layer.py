"""
From-scratch reimplementation of ONE Qwen2.5 decoder layer's forward pass.

This does NOT call HF's attention/MLP forward code -- pull raw weight
tensors out of an HF-loaded layer and reimplement every op (RMSNorm, RoPE,
GQA-aware causal attention, SwiGLU MLP) using plain torch ops.

SCOPE: single layer, single forward pass, no caching yet.
"""

import math
import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    TODO: unlike LayerNorm, no mean-subtraction, no bias.
    Formula: x / sqrt(mean(x^2, over last dim) + eps) * weight
    x: (batch, seq_len, hidden_size), weight: (hidden_size,)
    """
    rms = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return rms * weight


def build_rope_cache(seq_len: int, head_dim: int, base: float, device="cuda"):
    """
    TODO: RoPE precomputes cos/sin tables for positions [0, seq_len).
    1. inv_freq: one frequency per dimension PAIR -- shape (head_dim/2,).
       Formula: 1 / (base ** (2i / head_dim)) for i in range(head_dim/2).
    2. positions: 0..seq_len-1
    3. freqs = outer product of positions and inv_freq -> (seq_len, head_dim/2)
    4. concatenate freqs with itself along last dim -> (seq_len, head_dim)
    5. return freqs.cos(), freqs.sin()
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # shape (seq_len, head_dim/2)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()
    


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    TODO: split last dim in half, (x1, x2) -> (-x2, x1).
    This is what makes the cos/sin multiplication in apply_rope implement
    an actual rotation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1) 


def apply_rope(q, k, cos, sin):
    """
    TODO: q,k shape (batch, num_heads, seq_len, head_dim).
    cos,sin shape (seq_len, head_dim) -- need reshaping to broadcast over
    batch and heads dims before multiplying.
    Formula per tensor: (x * cos) + (rotate_half(x) * sin)
    """
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    TODO: GQA mechanic. x: (batch, num_kv_heads, seq_len, head_dim).
    Repeat each KV head n_rep times so it lines up with Q's head count.
    Return: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    Hint: expand + reshape, don't use a python loop.
    EDGE CASE: n_rep == 1 (no GQA, e.g. plain MHA) -- should be a no-op.
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def qwen_decoder_layer_forward(hidden_states: torch.Tensor, layer, config) -> torch.Tensor:
    """
    TODO: full forward pass for one decoder layer.

    layer is an HF Qwen2DecoderLayer -- use ONLY as a weight container:
        layer.input_layernorm.weight
        layer.self_attn.{q,k,v,o}_proj.weight / .bias
        layer.post_attention_layernorm.weight
        layer.mlp.{gate,up,down}_proj.weight
    Never call layer.forward() or layer.self_attn.forward().

    config gives: num_attention_heads, num_key_value_heads, hidden_size,
    rope_theta, rms_norm_eps.

    Steps, in order:
    1. residual = hidden_states
    2. x = rms_norm(hidden_states, input_layernorm.weight)
    3. project to q, k, v via F.linear (Qwen2.5 has bias on qkv -- pass it)
    4. reshape each to (batch, num_heads_or_kv_heads, seq_len, head_dim)
       -- note num_heads for q, num_kv_heads for k/v
    5. build rope cache, apply to q and k (NOT v)
    6. repeat_kv on k and v to match q's head count
    7. scaled dot-product: (q @ k^T) / sqrt(head_dim)
    8. add a causal mask (upper triangle = -inf, excluding diagonal)
    9. softmax, then attn_weights @ v
    10. reshape back to (batch, seq_len, hidden_size), project through o_proj
    11. hidden_states = residual + attn_output
    12. residual = hidden_states
    13. x = rms_norm(hidden_states, post_attention_layernorm.weight)
    14. SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
    15. hidden_states = residual + mlp_out
    16. return hidden_states
    """
    
    # Step 1: Residual
    residual = hidden_states

    # Step 2: RMSNorm
    x = rms_norm(residual, layer.input_layernorm.weight, eps=config.rms_norm_eps)

    # Step 3: Project to Q, K, V
    q_proj = F.linear(x, layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias)
    k_proj = F.linear(x, layer.self_attn.k_proj.weight, layer.self_attn.k_proj.bias)
    v_proj = F.linear(x, layer.self_attn.v_proj.weight, layer.self_attn.v_proj.bias)

    # Step 4: Reshape to (batch, num_heads_or_kv_heads, seq_len, head_dim)
    batch_size, seq_len, _ = q_proj.shape
    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_attention_heads
    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim).transpose(1, 2)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim).transpose(1, 2)
    v = v_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim).transpose(1, 2)

    # Step 5: Build RoPE cache and apply to Q and K
    cos, sin = build_rope_cache(seq_len, head_dim, config.rope_parameters["rope_theta"], device=x.device)
    q, k = apply_rope(q, k, cos, sin)

    # Step 6: Repeat KV heads to match Q's head count
    k = repeat_kv(k, config.num_attention_heads // config.num_key_value_heads)
    v = repeat_kv(v, config.num_attention_heads // config.num_key_value_heads)  

    # Step 7: Scaled dot-product attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Step 8: Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))   

    # Step 9: Softmax and attention output
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Step 10: Reshape back to (batch, seq_len, hidden_size) and project through output projection
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, layer.self_attn.o_proj.weight, layer.self_attn.o_proj.bias)

    # Step 11: Add residual
    hidden_states = residual + attn_output

    # Step 12: Residual for MLP
    residual = hidden_states

    # Step 13: RMSNorm before MLP
    x = rms_norm(hidden_states, layer.post_attention_layernorm.weight, eps=config.rms_norm_eps)

    # Step 14: SwiGLU MLP
    gate_proj = F.linear(x, layer.mlp.gate_proj.weight)
    up_proj = F.linear(x, layer.mlp.up_proj.weight)
    mlp_out = F.silu(gate_proj) * up_proj
    mlp_out = F.linear(mlp_out, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)

    # Step 15: Add residual
    hidden_states = residual + mlp_out

    # Step 16: Return final hidden states
    return hidden_states