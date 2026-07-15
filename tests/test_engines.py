"""
Correctness tests: every engine's greedy output must match a reference
exactly (for deterministic engines -- naive and kv_cache both qualify).

TODO(ruchit): implement these. Use a tiny model for test speed -- don't
load Qwen2.5-1.5B in a pytest run; use something like "sshleifer/tiny-gpt2"
or "hf-internal-testing/tiny-random-gpt2" so the test suite runs in seconds,
not minutes. Keep the actual Qwen2.5 models for bench/ scripts and
notebooks, not for pytest.

Structure to implement:

    import pytest
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inferlab.engines.naive import NaiveEngine
    from inferlab.engines.kv_cache import KVCacheEngine
    from inferlab.engines.base import GenerationConfig

    TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"  # verify this exists
                                                           # and is causal-LM
                                                           # compatible

    @pytest.fixture(scope="module")
    def tiny_model_and_tokenizer():
        ...load once, reuse across tests in this file...

    def test_kv_cache_matches_naive_greedy(tiny_model_and_tokenizer):
        # same prompt, do_sample=False, both engines
        # assert naive_result.generated_ids equals kv_cache_result.generated_ids
        # exactly (torch.equal, not allclose -- these are token ids)
        ...

    def test_naive_engine_handles_single_token_generation():
        # max_new_tokens=1 -- edge case, make sure step_timestamps has
        # exactly 2 entries (prefill + 1 decode) and doesn't crash
        ...

    def test_kv_cache_respects_eos_token():
        # force an eos_token_id that should trigger early, verify
        # generated_ids stops there and isn't padded/garbage past it
        ...

WHY THIS MATTERS: this file is your regression safety net for every future
engine. When paged_attention.py and batched.py show up in Week 2-3, you'll
add test_paged_matches_naive_greedy() etc. following the exact same
pattern -- get the pattern right and fast (tiny model) now.
"""