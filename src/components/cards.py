import streamlit as st

EMOTION_EMOJIS = {
    'Joy': '😊',
    'Anger': '😡',
    'Sadness': '😢',
    'Fear': '😨',
    'Surprise': '😲',
    'Disgust': '🤢',
    'Neutral': '😐'
}

def render_hero_prediction(emotion, confidence, model_name, inference_time):
    emoji = EMOTION_EMOJIS.get(emotion.capitalize(), '🤖')
    
    html = f"""
    <div class="st-key-hero_card" style="text-align: center;">
        <h1 style="font-size: 5rem; margin-bottom: 0;">{emoji}</h1>
        <h2 style="margin-top: 10px; color: var(--text-color);">{emotion.capitalize()}</h2>
        <h1 style="color: var(--primary-color); font-size: 3rem; margin: 10px 0;">{confidence:.1f}%</h1>
        <hr style="opacity: 0.2;">
        <div style="display: flex; justify-content: space-around; color: gray; font-size: 0.9rem;">
            <span><b>Model:</b> {model_name}</span>
            <span><b>Speed:</b> {inference_time:.1f} ms</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
def render_model_info(model_name, model_info):
    m_type = model_info.get("type", "Unknown").capitalize()
    params = model_info.get("params", "Unknown")
    speed = model_info.get("speed", "Unknown")
    
    html = f"""
    <div class="st-key-model_info">
        <h4 style="margin-top: 0;">🤖 {model_name} Info</h4>
        <ul style="color: var(--text-color); font-size: 0.95rem; line-height: 1.6;">
            <li><b>Architecture:</b> {m_type}</li>
            <li><b>Parameters:</b> {params}</li>
            <li><b>Speed:</b> {speed}</li>
        </ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
