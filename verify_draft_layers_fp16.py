import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferlab.attention.from_scratch_layer import qwen_decoder_layer_forward

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
model.eval()

prompt = "The quick brown fox jumps over the lazy dog and then"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    embed_out = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
    position_embeddings = model.model.rotary_emb(embed_out, position_ids)

    hf_hidden = embed_out
    scratch_hidden = embed_out

    for layer_idx, layer in enumerate(model.model.layers):
        hf_out = layer(hf_hidden, position_embeddings=position_embeddings)[0]
        if hf_out.dim() == 2:
            hf_out = hf_out.unsqueeze(0)

        scratch_out = qwen_decoder_layer_forward(scratch_hidden, layer, model.config, layer_idx, kv_cache=None)

        diff = (hf_out - scratch_out).abs().max().item()
        print(f"Layer {layer_idx}: diff={diff:.6f}, hf_max_abs={hf_out.abs().max().item():.4f}, scratch_max_abs={scratch_out.abs().max().item():.4f}")

        hf_hidden = hf_out
        scratch_hidden = scratch_out