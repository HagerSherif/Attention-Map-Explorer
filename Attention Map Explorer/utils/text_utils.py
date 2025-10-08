def encode_text(text,tokenizer):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return encoded["input_ids"]