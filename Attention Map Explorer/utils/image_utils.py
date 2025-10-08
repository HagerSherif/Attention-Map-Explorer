from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

def load_image(image,resize=224):
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def get_patch_grid(size=224, patch_size=16):
    patch_size = patch_size
    num_w = size // patch_size
    num_h = size // patch_size
    return patch_size, num_w, num_h

def draw_patch_box(image, patch_index, patch_size, num_w):
    img = image.resize((224, 224)).copy()
    draw = ImageDraw.Draw(img)
    px = (patch_index % num_w) * patch_size
    py = (patch_index // num_w) * patch_size
    draw.rectangle([px, py, px + patch_size, py + patch_size], outline="red", width=3)
    return img


def visualize_token_attention_grid(image_tensor, attn_maps, figsize=(50, 6)):
    """
    Row 0 of each row shows the token position in red.
    Columns 1 to N are attention overlays per layer.
    """
    num_layers = len(attn_maps)
    fig, axes = plt.subplots(1, num_layers , figsize=figsize, squeeze=False)
    for col, attn_map in enumerate(attn_maps):
        ax = axes[0, col ]

        # Undo normalization done at the beginning
        inv_transform = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        image = inv_transform(image_tensor).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        # Resize attention to image size
        attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
        attn_map = torch.nn.functional.interpolate(attn_map, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]

        # Plot image and overlay
        ax.imshow(image)
        ax.imshow(attn_map.detach().cpu().numpy(), cmap='jet', alpha=0.4)

        ax.axis("off")
        ax.set_title(f"L{col+1}", fontsize=25)
    return fig