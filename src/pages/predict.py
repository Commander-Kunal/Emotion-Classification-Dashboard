import streamlit as st
import numpy as np

from src.utils.model_manager import predict_emotion, EKMAN_LABELS, load_all_models
from src.components.cards import render_hero_prediction, render_model_info
from src.components.ranking import render_full_ranking
from src.components.charts import plot_horizontal_bar, plot_radar_chart, plot_pie_chart
import pandas as pd
from datetime import datetime

from src.utils.css import load_custom_css
load_custom_css()

st.title("🧠 Emotion Predictor")
st.markdown("Analyze text and predict emotions using the trained deep learning models.")

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# Load models
models_dict = load_all_models()

# Layout
col_main, col_side = st.columns([7, 3])

with col_side:
    st.markdown("### ⚙️ Settings")
    selected_model = st.selectbox("Select AI Model", list(models_dict.keys()) if models_dict else ["No models loaded"])
    
    if models_dict and selected_model in models_dict:
        render_model_info(selected_model, models_dict[selected_model])
        
    st.markdown("### 📊 Chart Type")
    chart_type = st.radio("Visualization", ["Horizontal Bar", "Radar Chart", "Pie Chart"])

with col_main:
    text_input = st.text_area("Enter text to analyze:", "I am absolutely thrilled about this new project!", height=150)
    
    if st.button("Analyze Emotion", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text.")
        elif not models_dict:
            st.error("❌ Models are not loaded yet.")
        else:
            with st.spinner(f"⏳ Running prediction using {selected_model}..."):
                probs, inf_time = predict_emotion(text_input, selected_model, models_dict)
                
            if probs is not None:
                st.success("✅ Prediction completed successfully.")
                
                # Hero Card
                pred_idx = np.argmax(probs)
                pred_emotion = EKMAN_LABELS[pred_idx]
                confidence = probs[pred_idx] * 100
                
                render_hero_prediction(pred_emotion, confidence, selected_model, inf_time)
                
                # Save to history
                st.session_state['prediction_history'].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Text": text_input,
                    "Model": selected_model,
                    "Emotion": pred_emotion.capitalize(),
                    "Confidence (%)": round(confidence, 1),
                    "Inference Time (ms)": round(inf_time, 1)
                })
                
                st.markdown("---")
                
                # Two column layout for Ranking and Chart
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    render_full_ranking(probs, EKMAN_LABELS)
                
                with res_col2:
                    st.markdown(f"### {chart_type}")
                    if chart_type == "Horizontal Bar":
                        st.plotly_chart(plot_horizontal_bar(probs, EKMAN_LABELS), use_container_width=True)
                    elif chart_type == "Radar Chart":
                        st.plotly_chart(plot_radar_chart(probs, EKMAN_LABELS), use_container_width=True)
                    else:
                        st.plotly_chart(plot_pie_chart(probs, EKMAN_LABELS), use_container_width=True)

# History Section at the bottom
st.markdown("---")
st.subheader("📜 Session History")
if st.session_state['prediction_history']:
    df_history = pd.DataFrame(st.session_state['prediction_history'])
    st.dataframe(df_history, use_container_width=True)
    
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("🗑️ Clear"):
            st.session_state['prediction_history'] = []
            st.rerun()
    with col2:
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name='emotion_predictions_history.csv',
            mime='text/csv',
        )
else:
    st.info("No predictions made in this session yet.")
