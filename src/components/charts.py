import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np

# Emotion to Color mapping (Consistent styling)
EMOTION_COLORS = {
    'Joy': '#2ecc71',
    'Anger': '#e74c3c',
    'Sadness': '#3498db',
    'Fear': '#9b59b6',
    'Surprise': '#f1c40f',
    'Disgust': '#e67e22',
    'Neutral': '#95a5a6'
}

def plot_horizontal_bar(probs, labels):
    df = pd.DataFrame({'Emotion': [l.capitalize() for l in labels], 'Confidence': probs * 100})
    df = df.sort_values(by='Confidence', ascending=True)
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Emotion', 
        orientation='h',
        color='Emotion',
        color_discrete_map=EMOTION_COLORS,
        text=df['Confidence'].apply(lambda x: f'{x:.1f}%')
    )
    fig.update_layout(
        showlegend=False, 
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_radar_chart(probs, labels):
    df = pd.DataFrame({'Emotion': [l.capitalize() for l in labels], 'Confidence': probs * 100})
    # Close the radar loop
    df = pd.concat([df, df.iloc[[0]]])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df['Confidence'],
        theta=df['Emotion'],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.4)',
        line=dict(color='#3498db', width=2),
        name='Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_pie_chart(probs, labels):
    df = pd.DataFrame({'Emotion': [l.capitalize() for l in labels], 'Confidence': probs * 100})
    fig = px.pie(
        df, 
        values='Confidence', 
        names='Emotion',
        color='Emotion',
        color_discrete_map=EMOTION_COLORS,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
