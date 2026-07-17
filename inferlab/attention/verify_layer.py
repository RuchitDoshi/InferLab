import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferlab.attention.from_scratch_layer import qwen_decoder_layer_forward


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to("cuda")
    model.eval()

    prompt = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        embed_out = model.model.embed_tokens(input_ids)

        # HF now computes RoPE cos/sin once, centrally, and passes them
        # into every layer -- rather than each layer computing it itself.
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        position_embeddings = model.model.rotary_emb(embed_out, position_ids)

    layer0 = model.model.layers[0]
    config = model.config

    with torch.no_grad():
        hf_output = layer0(embed_out, position_embeddings=position_embeddings)[0]

    with torch.no_grad():
        scratch_output = qwen_decoder_layer_forward(embed_out, layer0, config)

    max_abs_diff = (hf_output - scratch_output).abs().max().item()
    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"hf_output shape: {hf_output.shape}")
    print(f"scratch_output shape: {scratch_output.shape}")
    print(f"Scratch output sample: {scratch_output[0, 0, :5]}")

    assert torch.allclose(hf_output, scratch_output, atol=1e-3), (
        f"MISMATCH: max diff {max_abs_diff} exceeds tolerance. "
        f"Check RoPE base/convention, GQA repeat order, causal mask, "
        f"or whether q/k/v bias is being applied correctly."
    )
    print("PASS: from-scratch layer matches HF exactly (within tolerance).")


if __name__ == "__main__":
    main()