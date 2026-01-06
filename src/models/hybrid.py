"""Hybrid classifier combining transformer embeddings with hand-crafted features."""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/deberta-v3-base"
EMBEDDING_PATH = "models/hybrid_embeddings.pt"
CLASSIFIER_PATH = "models/hybrid_classifier.pkl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def get_embeddings(texts: list[str]) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

def train_hybrid(data_path: str = "data/processed/combined_prompts.parquet"):
    logger.info("Loading data for hybrid training...")
    df = pd.read_parquet(data_path)

    # Get transformer embeddings
    embeddings = get_embeddings(df["text"].tolist())
    torch.save(embeddings, EMBEDDING_PATH)

    # Load hand-crafted features (reuse from data module)
    from src.data.features import engineer_all
    from src.data.preprocessing import clean_text, normalize_labels
    from src.data.loaders import load_combined

    df_clean = clean_text(normalize_labels(load_combined()))
    features_df = engineer_all(df_clean)
    hand_features = features_df[["keyword_count", "special_char_ratio", "entropy", "char_length", "word_count"]].values

    # Concatenate
    X = torch.cat([embeddings, torch.tensor(hand_features, dtype=torch.float)], dim=1).numpy()
    y = df["label"].values

    # Train simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    joblib.dump(clf, CLASSIFIER_PATH)
    logger.info("Hybrid classifier trained and saved")