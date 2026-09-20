import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    /* Soft Card Styling for general use */
    .st-key-hero_card, .st-key-model_info, .st-key-history_card {
        background: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .st-key-hero_card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Metrics overriding to look like cards */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Global Typography Improvements */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Progress bars styling */
    .stProgress > div > div > div > div {
        border-radius: 10px;
    }
    
    /* Hide the top Streamlit header line */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
