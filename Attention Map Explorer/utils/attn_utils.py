def get_token_to_all_attn(attn_list, token_index=1,image_size=224, patch_size=16):
    """
    Extract attention from one token to all others across layers.
    Returns: list of [num_patches  x num_patches] attention maps (one (average over heads) per layer)
    """
    attn_maps = []
    num_patches=int(image_size /patch_size)
    for layer, attn_tensor in enumerate(attn_list):
        # attn_tensor: [1, num_heads, seq_len, seq_len]
        avg_attn = attn_tensor[0].mean(dim=0)  # [seq_len, seq_len]
        len=num_patches**2
        token_attn = avg_attn[token_index, 1:len+1]  # token → all patch tokens
        token_attn = token_attn.detach().cpu().numpy().reshape(num_patches,num_patches)
        attn_maps.append(token_attn)
    return attn_maps



def get_text_to_image_attn_all_layers(attn_list, text_token_idx=4, num_image_tokens=197):
    """
    Returns a list of attention vectors [197] from the given text token to image tokens,
    one per layer (mean over heads).
    """
    attn_maps = []
    for layer in range(len(attn_list)):
        attn = attn_list[layer][0]  # [num_heads, seq_len, seq_len]
        avg_attn = attn.mean(dim=0)  # [seq_len, seq_len]
        source_idx = num_image_tokens + text_token_idx
        attn_vector = avg_attn[source_idx, :num_image_tokens]  # [197]
        attn_maps.append(attn_vector.detach().cpu())
    return attn_maps  # list of tensors [197]|


