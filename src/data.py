"""Download the UCI SMS Spam Collection (5,574 real text messages, labelled ham/spam)."""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILE = DATA_DIR / "SMSSpamCollection"


def download() -> Path:
    if not RAW_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW_FILE} ...")
        with urlopen(URL) as resp, zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            RAW_FILE.write_bytes(zf.read("SMSSpamCollection"))
    return RAW_FILE


def load_data(drop_duplicates: bool = True) -> pd.DataFrame:
    # Tab separated: label \t message. quoting=3 (QUOTE_NONE) because messages contain stray quotes.
    df = pd.read_csv(download(), sep="\t", header=None, names=["label", "text"], quoting=3, encoding="utf-8")
    if drop_duplicates:
        # ~400 messages are exact repeats (mostly spam blasts). Leaving them in lets the
        # same text land in both train and test, which flatters the score.
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df
