import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
from src.utils.css import load_custom_css

st.set_page_config(page_title="Evaluation Metrics", page_icon="📊", layout="wide")
load_custom_css()

st.title("📊 Advanced Evaluation Dashboard")
st.markdown("Comprehensive analysis of model performance, speed, and dataset distribution.")

@st.cache_data
def load_metrics():
    try:
        with open(os.path.join("models", "metrics.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        return None

@st.cache_data
def load_labels():
    try:
        with open(os.path.join("models", "label_mappings.json"), "r") as f:
            mappings = json.load(f)
        return mappings['ekman_labels']
    except Exception:
        return ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

metrics_data = load_metrics()
ekman_labels = load_labels()

EMOTION_COLORS = {
    'Joy': '#2ecc71',
    'Anger': '#e74c3c',
    'Sadness': '#3498db',
    'Fear': '#9b59b6',
    'Surprise': '#f1c40f',
    'Disgust': '#e67e22',
    'Neutral': '#95a5a6'
}

if not metrics_data:
    st.error("❌ `metrics.json` not found in `models/` directory. Please run the Kaggle training script to generate it.")
else:
    # ---------------------------------------------------------
    # 1. Dataset & Efficiency Overview
    # ---------------------------------------------------------
    st.markdown("---")
    col_data, col_eff = st.columns(2)
    
    with col_data:
        st.subheader("📈 Dataset Distribution (GoEmotions Ekman)")
        # Representative distribution of GoEmotions dataset
        dist_data = {
            'Emotion': ['Neutral', 'Joy', 'Anger', 'Sadness', 'Surprise', 'Fear', 'Disgust'],
            'Count': [14000, 12000, 4500, 4000, 3000, 1500, 1000]
        }
        df_dist = pd.DataFrame(dist_data)
        fig_donut = px.pie(df_dist, values='Count', names='Emotion', hole=0.5, 
                           color='Emotion', color_discrete_map=EMOTION_COLORS)
        fig_donut.update_traces(textinfo='percent+label', textposition='outside')
        fig_donut.update_layout(showlegend=False, margin=dict(t=40, b=40, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_eff:
        st.subheader("⚡ Speed vs Accuracy (Efficiency)")
        eff_list = []
        for m in metrics_data:
            eff_list.append({
                "Model": m.get("Model"),
                "Macro F1": m.get("Macro F1", 0),
                "Inference Time (ms)": m.get("Inference Time per Batch (ms)", 1),
                "Model Size (MB)": m.get("Model Size (MB)", 1)
            })
        df_eff = pd.DataFrame(eff_list)
        
        fig_eff = px.scatter(df_eff, x="Inference Time (ms)", y="Macro F1", 
                             size="Model Size (MB)", color="Model", hover_name="Model",
                             size_max=40)
        fig_eff.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # Inverse X-axis so faster (lower ms) is on the right (better)
        fig_eff.update_xaxes(autorange="reversed", title="Inference Time per Batch (ms) ⬅️ Faster")
        fig_eff.update_yaxes(title="Macro F1 Score ⬆️ Better")
        st.plotly_chart(fig_eff, use_container_width=True)

    # ---------------------------------------------------------
    # 2. Per-Model Emotion Radar Charts
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🕸️ Model Strengths (Per-Class F1 Radar)")
    
    # Create columns dynamically based on number of models (max 3 per row)
    cols_per_row = 3
    for i in range(0, len(metrics_data), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(metrics_data):
                m = metrics_data[idx]
                model_name = m.get("Model")
                class_report = m.get("Per-Class Metrics")
                
                radar_emotions = []
                radar_scores = []
                if class_report:
                    for emo in ekman_labels:
                        if emo in class_report:
                            radar_emotions.append(emo.capitalize())
                            radar_scores.append(class_report[emo]['f1-score'])
                            
                # Close the loop
                if radar_emotions:
                    radar_emotions.append(radar_emotions[0])
                    radar_scores.append(radar_scores[0])
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_scores,
                        theta=radar_emotions,
                        fill='toself',
                        name=model_name,
                        line=dict(color='#3498db')
                    ))
                    fig_radar.update_layout(
                        title=dict(text=model_name, x=0.5, font=dict(size=14)),
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(visible=True, range=[0, 1.0]),
                            angularaxis=dict()
                        ),
                        showlegend=False,
                        margin=dict(t=40, b=20, l=40, r=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    cols[j].plotly_chart(fig_radar, use_container_width=True)

    # ---------------------------------------------------------
    # 3. Grouped Bar Chart (Per-Class F1-Score)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Per-Emotion F1-Score Comparison")
    
    per_class_data = []
    for m in metrics_data:
        model_name = m.get('Model')
        class_report = m.get('Per-Class Metrics')
        if class_report:
            for emotion in ekman_labels:
                if emotion in class_report:
                    per_class_data.append({
                        'Model': model_name,
                        'Emotion': emotion.capitalize(),
                        'F1-Score': class_report[emotion]['f1-score'],
                    })
                    
    if per_class_data:
        df_plot = pd.DataFrame(per_class_data)
        fig_bar = px.bar(df_plot, x='Emotion', y='F1-Score', color='Model', barmode='group',
                         text_auto='.2f')
        fig_bar.update_layout(margin=dict(t=20, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # 4. Learning Curves (Training History)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📉 Model Learning Curves")
    st.caption("Training Loss vs Validation F1. Gold Star (⭐) indicates the Best Epoch selected by Early Stopping.")
    
    models_with_history = [m for m in metrics_data if m.get("Training History")]
    if models_with_history:
        num_models = len(models_with_history)
        cols_count = 2
        rows_count = math.ceil(num_models / cols_count)
        
        fig_curves, axes = plt.subplots(rows_count, cols_count, figsize=(15, 6 * rows_count))
        fig_curves.patch.set_alpha(0.0)
        axes = axes.flatten() if num_models > 1 else [axes]
        
        for i, m in enumerate(models_with_history):
            model_name = m.get('Model')
            history = m.get('Training History', [])
            best_epoch = m.get('Best Epoch', -1)
            
            if len(history) == 0: continue
            
            epochs = [h['epoch'] for h in history]
            val_f1s = [h.get('val_f1', 0) for h in history]
            train_loss = [h.get('train_loss', 0) for h in history]
            
            ax1 = axes[i]
            ax1.patch.set_alpha(0.0)
            
            # Plot Validation F1 (Primary y-axis)
            color = '#3498db' # Light Blue
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Validation F1 Score', color=color)
            line1, = ax1.plot(epochs, val_f1s, marker='o', color=color, label='Val F1')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Highlight Best Epoch (Early Stopping Point)
            if best_epoch > 0:
                best_idx = epochs.index(best_epoch) if best_epoch in epochs else -1
                if best_idx != -1:
                    ax1.plot(epochs[best_idx], val_f1s[best_idx], marker='*', markersize=20, color='#f1c40f', markeredgecolor='black', label=f'Best Epoch ({best_epoch})')
                    ax1.annotate(f"{val_f1s[best_idx]:.3f}", 
                                 (epochs[best_idx], val_f1s[best_idx]), 
                                 textcoords="offset points", 
                                 xytext=(0,15), 
                                 ha='center', fontsize=10, fontweight='bold')

            # Plot Training Loss (Secondary y-axis)
            ax2 = ax1.twinx()
            color = '#e74c3c' # Red
            ax2.set_ylabel('Training Loss', color=color)
            line2, = ax2.plot(epochs, train_loss, marker='s', linestyle='--', color=color, alpha=0.6, label='Train Loss')
            ax2.tick_params(axis='y', labelcolor=color)
            
            ax1.set_title(f"{model_name}", fontsize=14, fontweight='bold')
            
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            handles, current_labels = ax1.get_legend_handles_labels()
            ax1.legend(handles + [line2], current_labels + ['Train Loss'], loc='center right')
            
        # Hide any empty subplots
        for j in range(i + 1, len(axes)):
            fig_curves.delaxes(axes[j])
            
        plt.tight_layout()
        st.pyplot(fig_curves)
