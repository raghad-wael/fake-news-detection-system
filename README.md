# Fake News Detection System

An end-to-end misinformation detection pipeline that cleans and merges 3 real-world datasets, benchmarks 9 models spanning classical ML to fine-tuned transformers, and deploys all the models as an interactive Streamlit web app — including a soft/hard voting ensemble comparison and live per-model stats (F1, MCC, latency, parameter count).

**Graduation Project — British University in Egypt (2025–2026)**

## Results

| Model | F1-Score | MCC | Est. Latency (CPU / T4 GPU) | Parameters |
|---|---|---|---|---|
| Naive Bayes | 84.40% | 0.683 | 0.32 ms / 0.31 ms | ~5K |
| Bi-LSTM | 95.16% | 0.901 | 0.83 ms / 0.83 ms | ~1.31M |
| DistilBERT | 97.29% | 0.945 | 20.15 ms / 3.80 ms | 66M |
| BERT | 97.64% | 0.952 | 40.88 ms / 7.39 ms | 110M |
| RoBERTa | 97.97% | 0.959 | 39.12 ms / 7.42 ms | 125M |
| Hard Voting Ensemble | 97.64% | 0.957 | 149.01 ms / 64.59 ms | ~302M |
| **Soft Voting Ensemble (best)** | **98.40%** | **0.968** | 149.01 ms / 64.59 ms | ~302M |

Logistic Regression, Random Forest, and Gradient Boosting were also trained as classical ML baselines (see notebook for full metrics).

## Datasets

Sourced from Kaggle and merged/deduplicated into a single training set:

- [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- [TextDB3 (Fake or Real News)](https://www.kaggle.com/datasets/hassanamin/textdb3)
- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

## Pipeline

1. **Cleaning** (`cleaning-datasets.ipynb`) — lowercases text, strips URLs/emails/HTML/mentions, removes non-letter characters, and applies TF-IDF + cosine-similarity deduplication (threshold 0.98) *within* each dataset individually to remove near-duplicate articles. This step alone removed 5,943 duplicates from ISOT (13.2%), 311 from TextDB3 (4.9%), and 9,638 from WELFake (13.4%).
2. **Training** (`01_data_pipeline_and_training.ipynb`) — merges the 3 cleaned datasets, then trains and evaluates 9 models: classical ML baselines (Naive Bayes, Logistic Regression, Random Forest, Gradient Boosting), a custom Bi-LSTM, and fine-tuned BERT/RoBERTa/DistilBERT (HuggingFace, PyTorch, AdamW with linear warmup).
3. **Ensembling & evaluation** (`02_evaluation_and_plots.ipynb`) — builds soft and hard voting ensembles from the trained models, and generates the comparison plots/metrics (F1, MCC with bootstrap confidence intervals, per-model latency).
4. **Deployment** (`streamlit_app.py`) — interactive app where the user selects a model from the sidebar (with live F1/MCC/latency/parameter stats shown), pastes in either raw article text or a URL (the app scrapes the article automatically), and gets a real-time real/fake classification.

## Tech stack

- Python, pandas, scikit-learn
- PyTorch, HuggingFace Transformers
- Streamlit, BeautifulSoup (URL article scraping)

## Repo contents

```
cleaning-datasets.ipynb              # raw dataset cleaning + deduplication
01_data_pipeline_and_training.ipynb  # dataset merge + model training (9 models)
02_evaluation_and_plots.ipynb        # ensembling, metrics, and evaluation plots
streamlit_app.py                     # deployed classification interface
checkpoints/                         # trained model weights (see below — not included in repo)
```

## Running the Streamlit app

The app expects a `checkpoints/` folder in the same directory as `streamlit_app.py`, containing the trained model artifacts. **Model checkpoints are not included in this repo** (too large for GitHub) — you'll need to run the training notebooks yourself to generate them, or contact me for the trained weights.

Expected structure:

```
checkpoints/
  tfidf_vectorizer.pkl              # shared TF-IDF vectorizer for all classical ML models
  logistic_regression.pkl
  random_forest.pkl
  gradient_boosting.pkl
  naive_bayes.pkl
  LSTM_final_weights.pth
  LSTM_vocabulary.pkl
  BERT_final_weights/               # HuggingFace model folder (config.json, model weights, tokenizer files)
  RoBERTa_final_weights/
  DistilBERT_final_weights/
```

Once the `checkpoints/` folder is in place:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will only show/enable models whose checkpoint files are found — missing files produce a clear in-app error rather than a crash.

## Note

Training and evaluation were run on Kaggle notebooks (GPU: Tesla T4); dataset paths reference `/kaggle/input/...` and will need adjusting for local runs.
