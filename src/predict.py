"""Classify a new message with the saved model.

Usage (from the project root):
    python src/predict.py "You won a free prize! Click now to claim."
    python src/predict.py            # interactive: type messages, empty line to quit
"""
import math
import os
import sys

import joblib

from preprocess import clean_text

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_cache = {}


def _load():
    if not _cache:
        model_path = os.path.join(ROOT, "models", "spam_model.pkl")
        vec_path = os.path.join(ROOT, "models", "vectorizer.pkl")
        if not (os.path.exists(model_path) and os.path.exists(vec_path)):
            raise FileNotFoundError("Saved model not found. Run `python src/train_model.py` first.")
        _cache["model"] = joblib.load(model_path)
        _cache["vec"] = joblib.load(vec_path)
    return _cache["model"], _cache["vec"]


def predict_message(message):
    """Return a dict: {'ok': bool, 'label': 'spam'|'ham', 'spam_probability': float, 'message': str}.

    Empty / whitespace-only input (or text that is empty after cleaning, e.g.
    only punctuation or stop words) returns ok=False with a clear message and no prediction.
    """
    if message is None or not str(message).strip():
        return {"ok": False, "label": None, "spam_probability": None,
                "message": "Please enter some text to classify."}
    model, vec = _load()
    cleaned = clean_text(message)
    if not cleaned:
        return {"ok": False, "label": None, "spam_probability": None,
                "message": "The message has no usable words after cleaning (only symbols, numbers "
                           "or very common words). Please enter a longer message."}
    X = vec.transform([cleaned])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:  # e.g. LinearSVC: squash the decision score into 0..1
        prob = 1 / (1 + math.exp(-float(model.decision_function(X)[0])))
    label = "spam" if int(model.predict(X)[0]) == 1 else "ham"
    return {"ok": True, "label": label, "spam_probability": prob, "message": ""}


def _show(msg):
    r = predict_message(msg)
    if not r["ok"]:
        print(r["message"])
    else:
        print(f"{r['label'].upper():4s} (spam probability {r['spam_probability']:.2f})  <- {msg}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _show(" ".join(sys.argv[1:]))
    else:
        print("Type a message (empty line to quit).")
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if not text.strip():
                break
            _show(text)
