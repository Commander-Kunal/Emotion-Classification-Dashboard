import streamlit as st
from src.utils.css import load_custom_css

from src.utils.css import load_custom_css
load_custom_css()

st.title("ℹ️ About the Emotion Classification AI")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📌 Project Overview
    This project is a comprehensive NLP pipeline that classifies text into 6 core Ekman emotions plus Neutral. 
    It demonstrates my ability to build, train, evaluate, and deploy Deep Learning models from scratch.
    
    ### 📊 Dataset
    - **Source:** Google Research `go_emotions`
    - **Original Classes:** 27 emotions
    - **Mapped Classes:** 6 Ekman Emotions + Neutral
    - **Imbalance Handling:** Balanced Class Weights
    
    ### 🚀 Technologies
    - **Deep Learning Framework:** PyTorch
    - **NLP Library:** Hugging Face Transformers
    - **Deployment:** Streamlit
    - **Visualizations:** Plotly & Seaborn
    """)

with col2:
    st.markdown("""
    ### 🧠 Supported Models
    I trained 5 different architectures to compare traditional Recurrent Neural Networks against modern Transformers:
    
    1. **RoBERTa:** State-of-the-art Transformer (125M params)
    2. **DistilBERT:** Lightweight, fast Transformer (66M params)
    3. **BiLSTM + Attention:** Custom Sequence Model with Attention (6M params)
    4. **LSTM:** Standard Long Short-Term Memory Network (5M params)
    5. **GRU:** Gated Recurrent Unit Network (4M params)
    
    ### 📈 Training Pipeline
    Models were trained utilizing early stopping, dynamic padding, and optimized dataloaders. 
    Transformers were fine-tuned using the Hugging Face Trainer API, while Sequence models were trained via custom PyTorch training loops.
    """)

st.markdown("---")
st.info("Developed by **Mostafa Abdallah**.")
