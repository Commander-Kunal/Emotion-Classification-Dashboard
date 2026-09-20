import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import numpy as np

from src.utils.model_manager import predict_emotion, EKMAN_LABELS, load_all_models
from src.components.ranking import EMOTION_EMOJIS

from src.utils.css import load_custom_css
load_custom_css()

st.title("⚖️ Model Comparison Mode")
st.markdown("Run simultaneous inference across all loaded models to find consensus.")

models_dict = load_all_models()

text_input = st.text_area("Enter text to analyze across all models:", "I am absolutely thrilled about this new project!", height=150)

if st.button("Run Simultaneous Inference", type="primary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    elif not models_dict:
        st.error("❌ Models are not loaded yet.")
    else:
        results = []
        with st.spinner("⏳ Running inference on all models simultaneously..."):
            for model_name in models_dict:
                probs, inf_time = predict_emotion(text_input, model_name, models_dict)
                if probs is not None:
                    pred_idx = np.argmax(probs)
                    emotion = EKMAN_LABELS[pred_idx].capitalize()
                    confidence = probs[pred_idx] * 100
                    results.append({
                        "Model": model_name,
                        "Prediction": emotion,
                        "Confidence": confidence,
                        "Inference Time (ms)": inf_time
                    })
                    
        if results:
            df = pd.DataFrame(results)
            
            # Analytics
            fastest_model = df.loc[df['Inference Time (ms)'].idxmin()]
            most_confident = df.loc[df['Confidence'].idxmax()]
            
            consensus_emotion = df['Prediction'].mode()[0]
            consensus_count = (df['Prediction'] == consensus_emotion).sum()
            consensus_emoji = EMOTION_EMOJIS.get(consensus_emotion, '')
            
            st.markdown("---")
            st.subheader("🏆 Comparison Analytics")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Majority Consensus", f"{consensus_emoji} {consensus_emotion}", f"{consensus_count}/{len(df)} Models")
            col2.metric("Highest Confidence", f"{most_confident['Model']}", f"{most_confident['Confidence']:.1f}%")
            col3.metric("Fastest Inference", f"{fastest_model['Model']}", f"{fastest_model['Inference Time (ms)']:.1f} ms")
            
            st.markdown("---")
            st.subheader("📈 Visual Comparison")
            
            c1, c2 = st.columns(2)
            with c1:
                fig_conf = px.bar(df, x='Model', y='Confidence', color='Prediction', 
                                  text=df['Confidence'].apply(lambda x: f'{x:.1f}%'),
                                  title="Prediction Confidence (%)")
                fig_conf.update_layout(margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_conf.update_traces(textposition='outside')
                fig_conf.update_yaxes(range=[0, 110])
                st.plotly_chart(fig_conf, use_container_width=True)
                
            with c2:
                # For inference time, lower is better. Let's color by model for variety.
                fig_time = px.bar(df, x='Model', y='Inference Time (ms)', color='Model',
                                  text=df['Inference Time (ms)'].apply(lambda x: f'{x:.1f} ms'),
                                  title="Inference Speed (ms) - Lower is Better")
                fig_time.update_layout(margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                fig_time.update_traces(textposition='outside')
                st.plotly_chart(fig_time, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Detailed Results Table")
            
            def highlight_best(s):
                is_max = s == s.max()
                is_min = s == s.min()
                if s.name == 'Confidence':
                    return ['color: #2ecc71; font-weight: bold' if v else '' for v in is_max]
                elif s.name == 'Inference Time (ms)':
                    return ['color: #2ecc71; font-weight: bold' if v else '' for v in is_min]
                else:
                    return ['' for _ in s]
                    
            styled_df = df.style.apply(highlight_best, subset=['Confidence', 'Inference Time (ms)']) \
                                .format({'Confidence': "{:.1f}%", 'Inference Time (ms)': "{:.1f}"}) \
                                .set_properties(**{'text-align': 'left'})
                                
            st.dataframe(styled_df, use_container_width=True)
