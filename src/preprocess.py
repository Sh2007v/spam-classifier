"""Text cleaning and data-loading helpers for the spam classifier.
 
NLTK is not required: scikit-learn ships an English stop-word list, so the
project runs with only the packages in requirements.txt.
"""
import re
import string
 
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 
LABEL_MAP = {"ham": 0, "spam": 1}
STOP_WORDS = set(ENGLISH_STOP_WORDS)
 
# Words that sklearn's stop-word list removes but that are strong spam signals
# ("you won", "your account", "click now", "call ..."). Removing them was found to
# hurt recall on obvious spam (see results/evaluation.md, "Improving the model").
# This is the "over-cleaning" pitfall from the project guidance.
KEEP_WORDS = frozenset({"you", "your", "yours", "now", "not", "no", "only", "never",
                        "please", "call", "we", "our", "us", "me", "my", "all"})
 
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
_NUMBER_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")
 
 
def clean_text(text, remove_numbers=True, remove_stop_words=True, keep_words=KEEP_WORDS):
    """Lowercase, strip punctuation/numbers/stop words and extra spaces.
 
    keep_words are exempt from stop-word removal (pass keep_words=() for plain
    removal of the whole stop-word list). Set remove_numbers=False to keep digits.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    if remove_numbers:
        text = _NUMBER_RE.sub(" ", text)
    tokens = _SPACE_RE.sub(" ", text).strip().split(" ")
    if remove_stop_words:
        stop = STOP_WORDS - set(keep_words)
        tokens = [t for t in tokens if t and t not in stop]
    else:
        tokens = [t for t in tokens if t]
    return " ".join(tokens)
 
 
def load_dataset(path):
    """Load a labeled spam dataset into a DataFrame with columns label, message.
 
    Supports two layouts:
      * CSV with a header row `label,message` (this project's dataset)
      * the original UCI "SMS Spam Collection" file: tab-separated, no header
    """
    # Kaggle's spam.csv is Windows-1252/latin-1 (the pound sign is a strong spam signal);
    # try UTF-8 first and fall back, so characters are not turned into "?".
    encoding = "utf-8"
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                first_line = f.readline()
                f.read()
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if "\t" in first_line and not first_line.lower().startswith("label"):
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding=encoding)
    else:
        df = pd.read_csv(path, skipinitialspace=True, encoding=encoding)
        df.columns = [c.strip().lower() for c in df.columns]
        # Kaggle's copy uses v1/v2 columns
        df = df.rename(columns={"v1": "label", "v2": "message", "text": "message"})
        df = df[["label", "message"]]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABEL_MAP)].copy()
    return df.reset_index(drop=True)
 
 
def prepare(df):
    """Add `clean` text and numeric `target` (ham=0, spam=1) columns."""
    df = df.copy()
    df["clean"] = df["message"].apply(clean_text)
    # plain stop-word removal, kept only to show the before/after effect in evaluation.md
    df["clean_basic"] = df["message"].apply(lambda t: clean_text(t, keep_words=()))
    df["target"] = df["label"].map(LABEL_MAP)
    return df
 