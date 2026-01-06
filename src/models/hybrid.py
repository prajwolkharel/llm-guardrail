"""Improved hybrid model: DeBERTa-v3 embeddings + hand-crafted features."""

import logging
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from pathlib import Path
from src.data.pipeline import run_pipeline
from src.data.constants import DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "microsoft/deberta-v3-base"
EMBEDDING_PATH = Path("models/hybrid_embeddings.pt")
CLASSIFIER_PATH = Path("models/hybrid_classifier.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Small batch to avoid OOM on 6GB GPU

# Lazy-loaded globals
_tokenizer = None
_model = None

def _load_deberta_model():
    """Lazy load DeBERTa model and tokenizer."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Loading DeBERTa-v3 model on {DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model

def get_deberta_embeddings(texts: list[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Extract mean-pooled embeddings in batches to avoid CUDA OOM."""
    tokenizer, model = _load_deberta_model()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings.cpu())

        # Clean up GPU memory
        del inputs, outputs, batch_embeddings
        torch.cuda.empty_cache()

    embeddings = torch.cat(all_embeddings, dim=0)
    logger.info(f"Extracted {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}")
    return embeddings

def train_hybrid(data_path: str = DATA_PATHS["processed"]):
    """Train the hybrid classifier."""
    logger.info("Starting hybrid model training...")

    # Ensure fresh processed data
    run_pipeline()

    logger.info("Loading processed data...")
    df = pd.read_parquet(data_path)

    logger.info("Extracting DeBERTa embeddings (batched)...")
    embeddings = get_deberta_embeddings(df["text"].tolist())
    
    EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, EMBEDDING_PATH)
    logger.info(f"Embeddings saved to {EMBEDDING_PATH}")

    logger.info("Preparing hand-crafted features...")
    from src.data.features import engineer_all
    from src.data.preprocessing import clean_text, normalize_labels
    from src.data.loaders import load_combined

    df_clean = clean_text(normalize_labels(load_combined()))
    features_df = engineer_all(df_clean)

    selected_features = ["char_length", "word_count", "keyword_count",
                         "special_char_ratio", "entropy"]
    hand_features = features_df[selected_features].values

    logger.info("Combining embeddings and hand-crafted features...")
    X = torch.cat([
        embeddings,
        torch.tensor(hand_features, dtype=torch.float)
    ], dim=1).numpy()

    y = df["label"].values

    logger.info("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X, y)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    logger.info(f"Hybrid classifier saved to {CLASSIFIER_PATH}")

    logger.info("Hybrid model training completed successfully!")

if __name__ == "__main__":
    train_hybrid()