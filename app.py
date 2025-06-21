# Streamlit Web App for Handwritten Digit Generation
# app.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🎨",
    layout="wide"
)

# Generator class
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

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    generator = Generator(latent_dim=100, img_size=28)

    try:
        checkpoint = torch.load("mnist_gan_model.pth", map_location=device)

        # Support both checkpoint styles
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)

        generator.to(device)
        generator.eval()
        return generator, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, device

# Generate digit images
def generate_digit_images(generator, device, digit, num_samples=5):
    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        generated_imgs = generator(noise, labels)
        imgs = generated_imgs.cpu().numpy()
        imgs = (imgs + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        return imgs

# Main Streamlit app
def main():
    st.title("🎨 Handwritten Digit Generator")
    st.markdown("""
    This app generates handwritten-style digit images using a Conditional GAN trained on MNIST.
    Select a digit and generate 5 unique variations.
    """)

    generator, device = load_model()

    st.sidebar.header("🎛️ Controls")
    digit = st.sidebar.selectbox("Select Digit (0–9)", list(range(10)))
    
    if st.sidebar.button("🎲 Generate 5 Images"):
        with st.spinner("Generating images..."):
            images = generate_digit_images(generator, device, digit, num_samples=5)
            st.session_state.generated_images = images
            st.session_state.selected_digit = digit

    if 'generated_images' in st.session_state:
        st.header(f"Generated Samples for Digit: {st.session_state.selected_digit}")
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                pil_img = Image.fromarray((img.squeeze() * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.resize((128, 128), Image.NEAREST)
                st.image(pil_img, caption=f"Sample {i+1}", use_container_width=True)

        st.subheader("💾 Download Images")
        download_cols = st.columns(5)
        for i, img in enumerate(st.session_state.generated_images):
            with download_cols[i]:
                pil_img = Image.fromarray((img.squeeze() * 255).astype(np.uint8), mode='L')
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                st.download_button(
                    label=f"📥 Sample {i+1}",
                    data=buffer.getvalue(),
                    file_name=f"digit_{st.session_state.selected_digit}_sample_{i+1}.png",
                    mime="image/png"
                )

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **Model:** Conditional GAN  
    **Dataset:** MNIST (28x28 grayscale)  
    **Framework:** PyTorch  
    **Features:** Generate digits 0–9, 5 variations  
    """)

    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Generator Architecture**  
        - Input: 100D noise + 10D one-hot label  
        - Layers: 256 → 512 → 1024 → 784 (28×28)  
        - Activations: LeakyReLU + BatchNorm  
        - Output: Tanh to scale images between -1 and 1  

        **Training Settings**  
        - Loss: Binary Cross Entropy  
        - Optimizer: Adam (lr=0.0002, β₁=0.5)  
        - Epochs: 50  
        - Trained on Google Colab (T4 GPU)  
        """)

if __name__ == "__main__":
    main()
