"""Inference utility."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .constants import MODEL_SAVE_PATH
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return {
        "benign_prob": probs[0][0].item(),
        "malicious_prob": probs[0][1].item(),
        "predicted_label": int(probs.argmax(dim=1).item())
    }