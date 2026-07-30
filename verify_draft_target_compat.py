from transformers import AutoConfig, AutoTokenizer

draft_name = "Qwen/Qwen2.5-0.5B"
target_name = "Qwen/Qwen2.5-1.5B"

# 1. vocab_size check
draft_config = AutoConfig.from_pretrained(draft_name)
target_config = AutoConfig.from_pretrained(target_name)
print("Draft vocab size:", draft_config.vocab_size)
print("Target vocab size:", target_config.vocab_size)
print("Vocab sizes match:", draft_config.vocab_size == target_config.vocab_size)    

# 2. tokenizer identity check
draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)
target_tokenizer = AutoTokenizer.from_pretrained(target_name)
text = "The quick brown fox jumps over the lazy dog and then ran into the forest to find its way home."

draft_input_ids = draft_tokenizer.encode(text, return_tensors="pt")
target_input_ids = target_tokenizer.encode(text, return_tensors="pt")
print("Draft input IDs:", draft_input_ids)
print("Target input IDs:", target_input_ids)
print("Input IDs match:", (draft_input_ids == target_input_ids).all())
