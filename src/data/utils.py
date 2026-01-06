"""Utility functions."""

import math
import re
import pandas as pd

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = pd.Series(list(text.lower())).value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in freq if p > 0)

def special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    special = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    return special / len(text)
