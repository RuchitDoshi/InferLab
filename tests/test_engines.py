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
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.engines.naive import NaiveEngine
from inferlab.engines.kv_cache import KVCacheEngine
from inferlab.engines.base import GenerationConfig

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    return model, tokenizer

def test_kv_cache_matches_naive_greedy(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    prompt = "Hello, world!"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Naive Engine
    naive_engine = NaiveEngine(model, device="cpu")  # CPU is fine for a tiny model
    naive_config = GenerationConfig(do_sample=False, max_new_tokens=5)
    naive_result = naive_engine.generate(input_ids, config=naive_config)
    
    # KV Cache Engine
    kv_cache_engine = KVCacheEngine(model, device="cpu")  # CPU is fine for a tiny model
    kv_cache_config = GenerationConfig(do_sample=False, max_new_tokens=5)
    kv_cache_result = kv_cache_engine.generate(input_ids, config=kv_cache_config)
    
    assert torch.equal(naive_result.generated_ids, kv_cache_result.generated_ids)


def test_naive_engine_handles_single_token_generation():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    prompt = "Hello"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    naive_engine = NaiveEngine(model, device="cpu")  # CPU is fine for a tiny model
    config = GenerationConfig(do_sample=False, max_new_tokens=1)
    result = naive_engine.generate(input_ids, config=config)
    
    assert len(result.step_timestamps) == 1  # 1 decode
    assert result.generated_ids.shape[1] == 1  # only one new token generated


def test_kv_cache_respects_eos_token():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    prompt = "Hello"
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    
    kv_cache_engine = KVCacheEngine(model, device="cpu")  # CPU is fine for a tiny model
    # First, find out what token the model actually generates first,
    # with no EOS constraint -- this makes the test deterministic
    # regardless of the tiny model's random weights.
    probe_config = GenerationConfig(do_sample=False, max_new_tokens=1)
    probe_result = kv_cache_engine.generate(input_ids, config=probe_config)
    forced_eos_id = probe_result.generated_ids[0, 0].item()

    # Now generate again, treating that exact token as EOS -- generation
    # must stop immediately at length 1.
    config = GenerationConfig(do_sample=False, max_new_tokens=10, eos_token_id=forced_eos_id)
    result = kv_cache_engine.generate(input_ids, config=config)

    assert result.generated_ids.shape[1] == 1
    assert result.generated_ids[0, 0].item() == forced_eos_id
