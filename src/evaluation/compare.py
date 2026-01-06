"""Compare baseline and hybrid models."""

import logging
import pandas as pd
from src.models.inference import predict as baseline_predict
from src.models.hybrid import get_deberta_embeddings, CLASSIFIER_PATH
from joblib import load
import torch
from sklearn.metrics import classification_report
from src.data.constants import DATA_PATHS
from .utils import save_comparison_report, plot_confusion_matrices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hybrid_classifier():
    embeddings = torch.load("models/hybrid_embeddings.pt")
    clf = load(CLASSIFIER_PATH)
    return embeddings, clf

def evaluate():
    df = pd.read_parquet(DATA_PATHS["processed"])
    texts = df["text"].tolist()
    y_true = df["label"].tolist()

    logger.info("Running baseline inference...")
    baseline_preds = [baseline_predict(text)["predicted_label"] for text in texts]
    baseline_report = classification_report(y_true, baseline_preds, output_dict=True)

    logger.info("Running hybrid inference...")
    embeddings, clf = load_hybrid_classifier()
    hand_features = df[["char_length", "word_count", "keyword_count", "special_char_ratio", "entropy"]].values
    X = torch.cat([embeddings, torch.tensor(hand_features, dtype=torch.float)], dim=1).numpy()
    hybrid_preds = clf.predict(X)
    hybrid_report = classification_report(y_true, hybrid_preds, output_dict=True)

    baseline_metrics = {
        "Accuracy": baseline_report["accuracy"],
        "Precision": baseline_report["1"]["precision"],
        "Recall": baseline_report["1"]["recall"],
        "F1": baseline_report["1"]["f1-score"],
    }

    hybrid_metrics = {
        "Accuracy": hybrid_report["accuracy"],
        "Precision": hybrid_report["1"]["precision"],
        "Recall": hybrid_report["1"]["recall"],
        "F1": hybrid_report["1"]["f1-score"],
    }

    logger.info("Saving comparison report...")
    save_comparison_report(baseline_metrics, hybrid_metrics)
    plot_confusion_matrices(y_true, baseline_preds, hybrid_preds)

if __name__ == "__main__":
    evaluate()