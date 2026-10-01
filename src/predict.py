"""Classify one or more messages from the command line.

Examples:
    python -m src.predict "WINNER!! You have been selected for a £1000 prize. Call 09061701461 now"
    python -m src.predict "hey are we still on for lunch tomorrow?"
"""
import argparse

import joblib

from src.data import MODELS_DIR
from src.preprocess import clean_series


def classify(messages):
    model = joblib.load(MODELS_DIR / "spam_model.joblib")
    proba = model.predict_proba(clean_series(messages))[:, 1]
    return [("SPAM" if p >= 0.5 else "ham", float(p)) for p in proba]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="+")
    args = parser.parse_args()
    for msg, (label, p) in zip(args.messages, classify(args.messages)):
        print(f"[{label:>4}  p(spam)={p:.3f}]  {msg}")


if __name__ == "__main__":
    main()
