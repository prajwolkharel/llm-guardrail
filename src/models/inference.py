"""Inference utility with lazy loading."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .constants import MODEL_SAVE_PATH
import torch

# Global variables — loaded only when first used
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        _model.eval()

def predict(text: str):
    _load_model()  # Load only when first prediction is made
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return {
        "benign_prob": probs[0][0].item(),
        "malicious_prob": probs[0][1].item(),
        "predicted_label": int(probs.argmax(dim=1).item())
    }