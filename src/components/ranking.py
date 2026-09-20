import streamlit as st
import numpy as np
import pandas as pd

EMOTION_EMOJIS = {
    'Joy': '😊',
    'Anger': '😡',
    'Sadness': '😢',
    'Fear': '😨',
    'Surprise': '😲',
    'Disgust': '🤢',
    'Neutral': '😐'
}

# Simple hex colors for progress bars (Streamlit uses these natively when possible)
# But actually st.progress doesn't support custom colors directly in the free API without hacks.
# We will use columns to build the ranking.
def render_full_ranking(probs, labels):
    st.markdown("### 🏅 Full Emotion Ranking")
    
    indices = np.argsort(probs)[::-1]
    
    medals = ["🥇 1st", "🥈 2nd", "🥉 3rd", "④ 4th", "⑤ 5th", "⑥ 6th", "⑦ 7th"]
    
    for rank, idx in enumerate(indices):
        emotion = labels[idx].capitalize()
        confidence = probs[idx]
        emoji = EMOTION_EMOJIS.get(emotion, '🤖')
        
        col1, col2, col3 = st.columns([2, 6, 2])
        with col1:
            st.markdown(f"**{medals[rank]}** &nbsp; {emoji} {emotion}")
        with col2:
            st.progress(float(confidence))
        with col3:
            st.markdown(f"**{confidence*100:.1f}%**")
