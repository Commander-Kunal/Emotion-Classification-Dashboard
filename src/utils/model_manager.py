import os
import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# ==========================================
# 1. Label Mapping & Preprocessing
# ==========================================
def load_assets():
    try:
        with open(os.path.join("models", "label_mappings.json"), "r") as f:
            mappings = json.load(f)
        return mappings['ekman_labels']
    except Exception:
        return ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

EKMAN_LABELS = load_assets()

def load_vocab():
    try:
        vocab = torch.load(os.path.join("models", "vocab.pth"), weights_only=False)
        return vocab
    except Exception:
        return None

def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def text_pipeline(text, vocab, max_len=50):
    pad_idx = vocab.get('<pad>', 0)
    unk_idx = vocab.get('<unk>', 1)
    tokens = [vocab.get(word, unk_idx) for word in tokenize(text)]
    if len(tokens) < max_len:
        tokens.extend([pad_idx] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
    return torch.tensor([tokens], dtype=torch.long) # Add batch dim

# ==========================================
# 2. PyTorch Sequence Models Definitions
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs):
        attn_weights = torch.softmax(self.attention(rnn_outputs), dim=1)
        context_vector = torch.sum(attn_weights * rnn_outputs, dim=1)
        return context_vector, attn_weights

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        rnn_outputs, _ = self.rnn(embedded)
        context_vector, attn_weights = self.attention(rnn_outputs)
        return self.fc(self.dropout(context_vector))

# ==========================================
# 3. Unified Model Loader
# ==========================================
@st.cache_resource(show_spinner="Loading models into memory...")
def load_all_models():
    models_dict = {}
    
    # 1. Load RoBERTa
    roberta_path = os.path.join("models", "saved_roberta-base")
    if os.path.exists(roberta_path):
        try:
            tok = AutoTokenizer.from_pretrained(roberta_path)
            mod = AutoModelForSequenceClassification.from_pretrained(roberta_path)
            mod.eval()
            models_dict["RoBERTa"] = {"type": "transformer", "tokenizer": tok, "model": mod, "params": "125M", "speed": "Fast"}
        except Exception as e:
            print(f"Failed to load RoBERTa: {e}")

    # 2. Load DistilBERT
    distilbert_path = os.path.join("models", "saved_distilbert-base-uncased")
    if os.path.exists(distilbert_path):
        try:
            tok = AutoTokenizer.from_pretrained(distilbert_path)
            mod = AutoModelForSequenceClassification.from_pretrained(distilbert_path)
            mod.eval()
            models_dict["DistilBERT"] = {"type": "transformer", "tokenizer": tok, "model": mod, "params": "66M", "speed": "Very Fast"}
        except Exception as e:
            print(f"Failed to load DistilBERT: {e}")

    # Load Vocab for Sequence Models
    vocab = load_vocab()
    if vocab:
        pad_idx = vocab.get('<pad>', 0)
        vocab_size = len(vocab)
        embed_dim = 100
        hidden_dim = 128
        output_dim = len(EKMAN_LABELS)
        
        # 3. Load LSTM
        lstm_path = os.path.join("models", "best_LSTM.pt")
        if os.path.exists(lstm_path):
            lstm = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, 2, True, 0.3, pad_idx)
            lstm.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu'), weights_only=True))
            lstm.eval()
            models_dict["LSTM"] = {"type": "sequence", "vocab": vocab, "model": lstm, "params": "5M", "speed": "Lightning"}

        # 4. Load GRU
        gru_path = os.path.join("models", "best_GRU.pt")
        if os.path.exists(gru_path):
            gru = GRUClassifier(vocab_size, embed_dim, hidden_dim, output_dim, 2, True, 0.3, pad_idx)
            gru.load_state_dict(torch.load(gru_path, map_location=torch.device('cpu'), weights_only=True))
            gru.eval()
            models_dict["GRU"] = {"type": "sequence", "vocab": vocab, "model": gru, "params": "4M", "speed": "Lightning"}

        # 5. Load BiLSTM + Attention
        bilstm_path = os.path.join("models", "best_BiLSTMAttention.pt")
        if os.path.exists(bilstm_path):
            bilstm = BiLSTMAttentionClassifier(vocab_size, embed_dim, hidden_dim, output_dim, 2, 0.3, pad_idx)
            bilstm.load_state_dict(torch.load(bilstm_path, map_location=torch.device('cpu'), weights_only=True))
            bilstm.eval()
            models_dict["BiLSTM + Attention"] = {"type": "sequence", "vocab": vocab, "model": bilstm, "params": "6M", "speed": "Lightning"}

    return models_dict

# ==========================================
# 4. Inference Function
# ==========================================
def predict_emotion(text, model_name, models_dict):
    import time
    start_time = time.time()
    
    if model_name not in models_dict:
        return None, 0, "Model not loaded"
        
    model_info = models_dict[model_name]
    m_type = model_info["type"]
    model = model_info["model"]
    
    with torch.no_grad():
        if m_type == "transformer":
            tokenizer = model_info["tokenizer"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
        elif m_type == "sequence":
            vocab = model_info["vocab"]
            inputs = text_pipeline(text, vocab)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)[0].numpy()
            
    inference_time = (time.time() - start_time) * 1000 # ms
    return probs, inference_time
