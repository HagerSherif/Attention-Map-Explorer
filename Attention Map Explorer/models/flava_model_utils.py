import torch
from models.flava_model import flava_model
from transformers import BertTokenizer
from utils.text_utils import encode_text

def flava_model_attn(image_tensor, text_input):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_tensor = encode_text(text_input, tokenizer)
    model = flava_model(pretrained=True)
    model.eval()

    with torch.no_grad():
        output = model(
            image=image_tensor,
            text=text_tensor,
            required_embedding="mm",
            skip_unmasked_mm_encoder=False
        )
    return output



