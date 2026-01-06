"""Dataset loaders."""

from datasets import load_dataset
import pandas as pd

def load_primary() -> pd.DataFrame:
    ds = load_dataset("deepset/prompt-injections")["train"]
    return pd.DataFrame(ds)

def load_secondary() -> pd.DataFrame:
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = pd.DataFrame(ds[split])
    df = df.rename(columns={"prompt": "text", "input": "text"}, errors="ignore")
    return df[["text", "label"]]

def load_combined() -> pd.DataFrame:
    df1 = load_primary()
    df2 = load_secondary()
    return pd.concat([df1[["text", "label"]], df2[["text", "label"]]], ignore_index=True)