"""Baseline model training."""

import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from .constants import MODEL_NAME, MODEL_SAVE_PATH, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DATA_PATH
from .utils import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_baseline():
    logger.info("Loading processed data...")
    df = pd.read_parquet(DATA_PATH)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(df):
        return tokenizer(df["text"].tolist(), padding="max_length", truncation=True, max_length=256)

    train_enc = tokenize(train_df)
    test_enc = tokenize(test_df)

    train_dataset = Dataset.from_dict({
        "input_ids": train_enc["input_ids"],
        "attention_mask": train_enc["attention_mask"],
        "labels": train_df["label"].tolist()
    })

    test_dataset = Dataset.from_dict({
        "input_ids": test_enc["input_ids"],
        "attention_mask": test_enc["attention_mask"],
        "labels": test_df["label"].tolist()
    })

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",           # ← changed
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # ← changed
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logger.info(f"Test metrics: {metrics}")

    logger.info(f"Saving model to {MODEL_SAVE_PATH}")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_baseline()