import streamlit as st
from src.components.sidebar import render_sidebar
from src.utils.model_manager import load_all_models

# Must be the first Streamlit command
st.set_page_config(page_title="Emotion Classification", page_icon="🧠", layout="wide")

# Preload models at the application root so they are ready for any page
models_dict = load_all_models()

# Render the global sidebar
render_sidebar(models_dict)

# Define Pages
page_predict = st.Page("src/pages/predict.py", title="Predict Emotion", icon="🎯", default=True)
page_compare = st.Page("src/pages/compare.py", title="Compare Models", icon="⚖️")
page_evaluation = st.Page("src/pages/evaluation.py", title="Evaluation Metrics", icon="📊")
page_about = st.Page("src/pages/about.py", title="About Project", icon="ℹ️")

# Build Navigation
pg = st.navigation({
    "Model Inference": [page_predict, page_compare],
    "Information": [page_evaluation, page_about]
})

# Run the selected page
pg.run()
