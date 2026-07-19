import streamlit as st
import torch
import torch.nn as nn
import re
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# ==================================================
# BASE PATH 
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# SESSION STATE INIT 
# ==================================================
if "title" not in st.session_state:
    st.session_state.title = ""

if "text" not in st.session_state:
    st.session_state.text = ""

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ==================================================
# CLEAN TEXT
# ==================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r'http\S+|www\S+|https\S+', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'@\w+|#\w+', ' ', txt)
    txt = re.sub(r'[^a-zA-Z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None, None

        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.text if soup.title else "No title"

        text = ""
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])
        
        if not text:
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])

        return title, text.strip()
    except Exception:
        return None, None

# ==================================================
# PYTORCH LSTM CLASS DEFINITION
# ==================================================
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier required to load the .pth file"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                   batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

# ==================================================
# MODEL LOADER
# ==================================================
@st.cache_resource
def load_model(selected_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    processor = None
    model_type = None

    try:
        checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
        
        ml_filename_map = {
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
            "Naive Bayes": "naive_bayes.pkl"
        }

        # ---------------- ML MODELS ----------------
        if selected_model in ml_filename_map:
            model_type = "sklearn"
            vec_path = os.path.join(checkpoint_dir, "tfidf_vectorizer.pkl")
            if os.path.exists(vec_path): processor = joblib.load(vec_path)
            else:
                st.error("TF-IDF Vectorizer not found!")
                return None, None, None, None

            model_file = ml_filename_map[selected_model]
            model_path = os.path.join(checkpoint_dir, model_file)
            if os.path.exists(model_path): model = joblib.load(model_path)
            else: st.error(f"ML model file '{model_file}' not found.")

        # ---------------- BI-LSTM MODEL ----------------
        elif selected_model == "Bi-LSTM":
            model_type = "lstm"
            model_path = os.path.join(checkpoint_dir, "LSTM_final_weights.pth")
            vocab_path = os.path.join(checkpoint_dir, "LSTM_vocabulary.pkl")

            if os.path.exists(model_path) and os.path.exists(vocab_path):
                # Load the dictionary mapping words to integers
                processor = joblib.load(vocab_path) 
                
                # Initialize architecture and load weights
                model = LSTMClassifier(vocab_size=len(processor))
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
            else:
                st.error("LSTM_final_weights.pth OR LSTM_vocabulary.pkl is missing in checkpoints folder.")

        # ---------------- TRANSFORMERS ----------------
        else:
            model_type = "transformer"
            model_path = os.path.join(checkpoint_dir, f"{selected_model}_final_weights")
            if os.path.exists(model_path):
                processor = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
            else:
                st.error(f"Transformer folder '{selected_model}_final_weights' missing.")

        return model, processor, model_type, device

    except Exception as e:
        st.error(f"Loading error: {e}")
        return None, None, None, None

# ==================================================
# SINGLE PREDICTION HELPER
# ==================================================
def get_single_prediction(full_text, model, processor, model_type, device):
    
    if model_type == "transformer":
        inputs = processor(full_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            prob_real = probs[0][0].item() * 100
            prob_fake = probs[0][1].item() * 100
            pred = 1 if prob_fake > prob_real else 0
            
    elif model_type == "lstm":
        max_len = 256
        vocab = processor # For LSTM, processor is our word_to_idx dictionary
        words = full_text.split()[:max_len]
        
        # Convert words to ints, default to 0 (<PAD>/Unknown) if not in vocab
        seq = [vocab.get(word, 0) for word in words]
        length = min(len(seq), max_len)
        
        # Prevent empty sequence crash
        if length == 0: 
            length = 1
            seq = [0]
            
        seq = seq + [0] * (max_len - len(seq)) # Pad to 256
        
        seq_tensor = torch.LongTensor([seq]).to(device)
        len_tensor = torch.LongTensor([length])
        
        with torch.no_grad():
            outputs = model(seq_tensor, len_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            prob_real = probs[0][0].item() * 100
            prob_fake = probs[0][1].item() * 100
            pred = 1 if prob_fake > prob_real else 0

    else: # Sklearn
        vec = processor.transform([full_text])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            prob_real = probs[0] * 100
            prob_fake = probs[1] * 100
        else:
            pred = model.predict(vec)[0]
            prob_fake = 100 if pred == 1 else 0
            prob_real = 100 if pred == 0 else 0
        pred = 1 if prob_fake > prob_real else 0
        
    return prob_fake, prob_real, pred

# ==================================================
# UI
# ==================================================
st.title("📰 AI Fake News Detection System")
st.sidebar.header("Models")

# 🌟 ADDED BI-LSTM TO CHOICES
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Soft Voting Ensemble", "Hard Voting Ensemble", 
     "DistilBERT", "RoBERTa", "BERT", "Bi-LSTM",
     "Logistic Regression", "Random Forest",
     "Gradient Boosting", "Naive Bayes"]
)

# 🌟 ADDED BI-LSTM TO ENSEMBLE BASE (Total 8 models now)
ENSEMBLE_BASE_MODELS = [
    "DistilBERT", "RoBERTa", "BERT", "Bi-LSTM",
    "Logistic Regression", "Random Forest", 
    "Gradient Boosting", "Naive Bayes"
]

is_ensemble = "Voting" in model_choice
if not is_ensemble:
    model, processor, model_type, device = load_model(model_choice)
    if model:
        st.sidebar.success(f"{model_choice} Loaded")
else:
    st.sidebar.info(f"{model_choice} Selected. Base models will load during analysis.")

# Model Specifications
st.sidebar.divider()
st.sidebar.subheader("Model Specifications")
model_specs = {
    "Logistic Regression": {"f1": "94.66%", "mcc": "0.891", "speed": "0.20 ms (CPU) | 0.14 ms (T4 GPU)", "params": "~5,000", "type": "Classical ML"},
    "Gradient Boosting": {"f1": "95.10%", "mcc": "0.900", "speed": "0.41 ms (CPU) | 0.35 ms (T4 GPU)", "params": "100 Estimators", "type": "Ensemble ML"},
    "Random Forest": {"f1": "94.51%", "mcc": "0.888", "speed": "47.10 ms (CPU) | 44.35 ms (T4 GPU)", "params": "200 Estimators", "type": "Ensemble ML"},
    "Naive Bayes": {"f1": "84.40%", "mcc": "0.683", "speed": "0.32 ms (CPU) | 0.31 ms (T4 GPU)", "params": "~5,000", "type": "Classical ML"},
    "Bi-LSTM": {"f1": "95.16%", "mcc": "0.901", "speed": "0.83 ms (CPU/GPU)", "params": "~1.31 Million", "type": "Recurrent Neural Net"},
    "DistilBERT": {"f1": "97.29%", "mcc": "0.945", "speed": "20.15 ms (CPU) | 3.80 ms (T4 GPU)", "params": "66 Million", "type": "Transformer"},
    "BERT": {"f1": "97.64%", "mcc": "0.952", "speed": "40.88 ms (CPU) | 7.39 ms (T4 GPU)", "params": "110 Million", "type": "Transformer"},
    "RoBERTa": {"f1": "97.97%", "mcc": "0.959", "speed": "39.12 ms (CPU) | 7.42 ms (T4 GPU)", "params": "125 Million", "type": "Transformer"},
    "Soft Voting Ensemble": {"f1": "98.40%", "mcc": "0.968", "speed": "149.01 ms (CPU) | 64.59 ms (T4 GPU)", "params": "~302 Million", "type": "Hybrid Aggregation"},
    "Hard Voting Ensemble": {"f1": "97.64%", "mcc": "0.957", "speed": "149.01 ms (CPU) | 64.59 ms (T4 GPU)", "params": "~302 Million", "type": "Hybrid Majority"}
}

if model_choice in model_specs:
    specs = model_specs[model_choice]
    st.sidebar.write(f"**Type:** {specs['type']}")
    st.sidebar.write(f"**F1-Score:** {specs['f1']}")
    st.sidebar.write(f"**MCC:** {specs['mcc']}")
    st.sidebar.write(f"**Parameters:** {specs['params']}")
    st.sidebar.write(f"**Est. Latency:** {specs['speed']}")

# ==================================================
# INPUT SECTION
# ==================================================
st.subheader("Input News")

input_type = st.radio("Input Type", ["URL", "Text"])
title = ""
text = ""

if input_type == "URL":
    url = st.text_input("Enter URL")
    if st.button("Fetch Article"):
        if url:
            with st.spinner("Fetching article..."):
                title, text = extract_article(url)

            if text:
                st.session_state.title = title
                st.session_state.text = text
                st.success("Article fetched successfully")
            else:
                st.error("Failed to extract article. The site may be using dynamic JavaScript or anti-bot protections.")

    if st.session_state.text:
        st.markdown("### Title")
        st.write(st.session_state.title)
        st.markdown("### Full Article")
        st.text_area("", st.session_state.text, height=300)

else:
    title = st.text_input("Title")
    text = st.text_area("Article Text", height=200)

# ==================================================
# PREDICTION 
# ==================================================
if st.button("Analyze"):
    if input_type == "URL":
        text = st.session_state.text
        title = st.session_state.title

    if not is_ensemble and not model:
        st.error("Model not loaded")
        st.stop()

    if not text:
        st.warning("Please provide text")
        st.stop()

    full_text = clean_text(f"{title} {text}")
    if len(full_text) < 20:
        st.warning("Text too short")
        st.stop()

    prob_fake = 0
    prob_real = 0

    if is_ensemble:
        st.info(f"Executing {model_choice}... This requires running all {len(ENSEMBLE_BASE_MODELS)} base models.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_fake_probs = []
        all_preds = []
        
        for i, m_name in enumerate(ENSEMBLE_BASE_MODELS):
            status_text.text(f"Running {m_name}...")
            m, p, t_type, d = load_model(m_name)
            
            if m is None:
                st.error(f"Missing files for {m_name}. Ensemble aborted.")
                st.stop()
                
            pf, pr, p_class = get_single_prediction(full_text, m, p, t_type, d)
            all_fake_probs.append(pf)
            all_preds.append(p_class)
            
            progress_bar.progress((i + 1) / len(ENSEMBLE_BASE_MODELS))
            
        status_text.text("Ensemble computation complete!")
        
        if model_choice == "Soft Voting Ensemble":
            prob_fake = np.mean(all_fake_probs)
            prob_real = 100 - prob_fake
            
        elif model_choice == "Hard Voting Ensemble":
            fake_votes = sum(all_preds)
            total_votes = len(all_preds)
            prob_fake = (fake_votes / total_votes) * 100
            prob_real = 100 - prob_fake

    else:
        with st.spinner(f"Analyzing with {model_choice}..."):
            prob_fake, prob_real, _ = get_single_prediction(full_text, model, processor, model_type, device)

    # ==================================================
    # OUTPUT
    # ==================================================
    st.divider()
    st.subheader("Result")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Real News: {prob_real:.2f}%**")
        st.progress(int(prob_real) if not np.isnan(prob_real) else 0)
    with col2:
        st.markdown(f"**Fake News: {prob_fake:.2f}%**")
        st.progress(int(prob_fake) if not np.isnan(prob_fake) else 0)

    st.write("")
    if prob_fake > prob_real:
        st.error(f"🚨 FAKE NEWS ({prob_fake:.2f}%)")
        if is_ensemble and model_choice == "Hard Voting Ensemble":
            st.error(f"*(Majority Vote: {sum(all_preds)} out of {len(all_preds)} models flagged this as fake)*")
    else:
        st.success(f"✅ REAL NEWS ({prob_real:.2f}%)")
        if is_ensemble and model_choice == "Hard Voting Ensemble":
            st.success(f"*(Majority Vote: {len(all_preds) - sum(all_preds)} out of {len(all_preds)} models flagged this as real)*")