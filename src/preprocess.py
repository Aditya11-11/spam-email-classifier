"""Text cleaning for spam detection.

A few choices worth explaining:
- URLs, phone numbers, money amounts and plain numbers get replaced with
  placeholder tokens. The *exact* number doesn't matter, but "this message
  contains a phone number and a £ amount" is a huge spam signal.
- Stopwords are removed using scikit-learn's built-in English list (no extra
  downloads needed).
- Porter stemming folds "winning", "winner", "wins" together into "win".
"""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|co\.uk|net|org)\S*)", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MONEY_RE = re.compile(r"[£$€]\s?\d+([.,]\d+)?|\d+([.,]\d+)?\s?(p|pence|pounds?)\b", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
NON_ALPHA_RE = re.compile(r"[^a-z_\s]")


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def clean_text(text: str, stem: bool = True, remove_stopwords: bool = True) -> str:
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    if stem:
        words = [_stem(w) for w in words]
    return " ".join(words)


def clean_series(texts):
    return [clean_text(t) for t in texts]
