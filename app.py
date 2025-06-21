# app.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🧠",
    layout="wide"
)

# --- Generator Model ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels_onehot = torch.zeros(labels.size(0), 10).to(labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        gen_input = torch.cat([noise, labels_onehot], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

# --- Load Trained Model ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    generator = Generator(latent_dim=100, img_size=28)
    try:
        checkpoint = torch.load("mnist_gan_model.pth", map_location=device)
        if isinstance(checkpoint, dict):
            if "generator_state_dict" in checkpoint:
                generator.load_state_dict(checkpoint["generator_state_dict"])
            else:
                generator.load_state_dict(checkpoint)
        else:
            generator.load_state_dict(checkpoint)
        generator.to(device)
        generator.eval()
        st.success("✅ Model loaded successfully.")
        return generator, device
    except Exception as e:
        st.error(f"❌ Failed to load model:\n\n**{e}**")
        return None, device

# --- Generate Images ---
def generate_digit_images(generator, device, digit, num_samples=5):
    if generator is None:
        st.warning("Model not loaded properly. Generating blank images.")
        return [np.zeros((1, 28, 28))] * num_samples

    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        generated_imgs = generator(noise, labels)
        imgs = generated_imgs.cpu().numpy()
        imgs = (imgs + 1) / 2  # Normalize to [0, 1]
        return imgs

# --- App Main ---
def main():
    st.title("🧠 Handwritten Digit Generator")
    st.markdown("""
    Generate handwritten-style digits using a conditional GAN trained on MNIST.  
    Select a digit from 0–9 and generate 5 unique samples.
    """)

    generator, device = load_model()

    # Sidebar Inputs
    st.sidebar.header("🎛️ Controls")
    digit = st.sidebar.selectbox("Select Digit", list(range(10)), index=0)
    generate = st.sidebar.button("🎲 Generate 5 Images")

    if generate:
        st.session_state.generated_images = generate_digit_images(generator, device, digit)
        st.session_state.generated_digit = digit

    # Display Images
    if 'generated_images' in st.session_state:
        st.header(f"Generated Samples for Digit: {st.session_state.generated_digit}")
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                img_display = (img.squeeze() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_display, mode='L')
                st.image(pil_img.resize((128, 128)), caption=f"Sample {i+1}", use_column_width=True)

        # Download Buttons
        st.subheader("💾 Download Each Image")
        cols_dl = st.columns(5)
        for i, img in enumerate(st.session_state.generated_images):
            img_display = (img.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_display, mode='L')
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            with cols_dl[i]:
                st.download_button(
                    f"Download {i+1}",
                    data=byte_im,
                    file_name=f"digit_{digit}_sample_{i+1}.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
