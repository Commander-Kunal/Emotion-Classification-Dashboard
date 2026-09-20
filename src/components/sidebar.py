import streamlit as st
import torch
import transformers

def render_sidebar(models_dict):
    st.sidebar.markdown("### ⚙️ System Status")
    
    # Status Indicator
    if len(models_dict) > 0:
        st.sidebar.success(f"🟢 {len(models_dict)} Models Loaded & Ready")
    else:
        st.sidebar.error("🔴 Models Not Loaded")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 Environment")
    device = "CUDA 🚀" if torch.cuda.is_available() else "CPU 🐢"
    st.sidebar.info(f"**Device:** {device}")
    st.sidebar.text(f"PyTorch: {torch.__version__}")
    st.sidebar.text(f"Transformers: {transformers.__version__}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.caption("Emotion Classification System developed with Streamlit, PyTorch, and Hugging Face Transformers.")
    st.sidebar.caption("Developed by **Mostafa Abdallah**")
