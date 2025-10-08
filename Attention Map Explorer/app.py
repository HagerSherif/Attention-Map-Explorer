import streamlit as st
from PIL import Image
from models.flava_model_utils import flava_model_attn
from utils.image_utils import load_image, draw_patch_box, get_patch_grid, visualize_token_attention_grid
from utils.attn_utils import get_token_to_all_attn

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Multimodal Attention Explorer")

# === Sidebar ===
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select a model", ["FLAVA"])  # Add CLIP, etc. later

# === Main UI ===
st.title("Multimodal Attention Map Explorer")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    patch_size, num_w, num_h = get_patch_grid(size=224, patch_size=16)
    max_index = num_w * num_h 
    patch_index = st.slider("Select patch index", 1, max_index, 1)

    image_with_box = draw_patch_box(image, patch_index - 1, patch_size, num_w)
    st.image(image_with_box, caption=f"Selected Patch: {patch_index}")

    if st.button("Run Attention!"):
        img_tensor = load_image(image,resize=224)
        output = flava_model_attn(img_tensor, " ")  # dummy text
        image_attentions=get_token_to_all_attn(output.image_attentions, token_index=patch_index,image_size=224, patch_size=16)
        fig = visualize_token_attention_grid(img_tensor[0], image_attentions)
        st.pyplot(fig)
