"""Data cleaning and normalization."""

import pandas as pd

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    return df

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].astype(int)
    return df