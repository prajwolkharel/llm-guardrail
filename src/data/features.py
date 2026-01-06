"""Feature engineering."""

import pandas as pd
from .constants import KEYWORDS
from .utils import special_char_ratio, calculate_entropy

def engineer_basic(df: pd.DataFrame) -> pd.DataFrame:
    df["char_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df

def engineer_advanced(df: pd.DataFrame) -> pd.DataFrame:
    df["keyword_count"] = df["text"].str.lower().apply(
        lambda x: sum(kw in x for kw in KEYWORDS)
    )
    df["special_char_ratio"] = df["text"].apply(special_char_ratio)
    df["entropy"] = df["text"].apply(calculate_entropy)
    return df

def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_advanced(engineer_basic(df))