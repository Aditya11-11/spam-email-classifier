"""What does spam look like compared to normal messages?

Run:  python -m src.eda
"""
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import FIGURES_DIR, load_data
from src.preprocess import clean_text


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    raw = load_data(drop_duplicates=False)
    df = load_data()
    print(f"Raw messages: {len(raw)}  |  after removing duplicates: {len(df)}")
    print(df["label"].value_counts())

    df["length"] = df["text"].str.len()
    df["digits"] = df["text"].str.count(r"\d")
    df["caps_ratio"] = df["text"].apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1))
    print("\nMedian stats by label:\n", df.groupby("label")[["length", "digits", "caps_ratio"]].median().round(3))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["length", "digits", "caps_ratio"],
                              ["Message length (chars)", "Number of digits", "Share of CAPITAL letters"]):
        sns.histplot(data=df, x=col, hue="label", bins=40, stat="density", common_norm=False,
                     palette={"ham": "#4C72B0", "spam": "#C44E52"}, ax=ax, element="step")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "message_stats.png", dpi=120)
    plt.close(fig)

    # Most common tokens in each class after cleaning
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, color in zip(axes, ["ham", "spam"], ["#4C72B0", "#C44E52"]):
        counts = Counter(w for t in df.loc[df["label"] == label, "text"] for w in clean_text(t).split())
        words, freqs = zip(*counts.most_common(20))
        ax.barh(words[::-1], freqs[::-1], color=color)
        ax.set_title(f"Top 20 tokens in {label.upper()}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_tokens.png", dpi=120)
    plt.close(fig)
    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
