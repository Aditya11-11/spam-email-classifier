"""Bag-of-Words vs TF-IDF, each with Multinomial Naive Bayes.

For a spam filter, the expensive mistake is a false positive: sending your
friend's message to the spam folder. So besides F1, I care most about spam
*precision* — when the model says "spam", it had better be right.

Run:  python -m src.train
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, load_data
from src.preprocess import clean_series

RANDOM_STATE = 42


def make_searches(cv):
    grid = {
        "vec__ngram_range": [(1, 1), (1, 2)],
        "vec__min_df": [1, 2],
        "nb__alpha": [0.01, 0.05, 0.1, 0.3, 1.0],
    }
    return {
        "Bag-of-Words + NB": GridSearchCV(
            Pipeline([("vec", CountVectorizer()), ("nb", MultinomialNB())]), grid, cv=cv, scoring="f1", n_jobs=-1),
        "TF-IDF + NB": GridSearchCV(
            Pipeline([("vec", TfidfVectorizer(sublinear_tf=True)), ("nb", MultinomialNB())]),
            grid, cv=cv, scoring="f1", n_jobs=-1),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df["clean"] = clean_series(df["text"])
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["clean"], df["is_spam"], df["text"], test_size=0.2, stratify=df["is_spam"], random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, search in make_searches(cv).items():
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        proba = search.predict_proba(X_test)[:, 1]
        results[name] = {
            "best_params": {k.split("__")[1]: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
            "cv_f1": round(float(search.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_spam_precision": round(float(precision_score(y_test, pred)), 4),
            "test_spam_recall": round(float(recall_score(y_test, pred)), 4),
            "test_spam_f1": round(float(f1_score(y_test, pred)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "vocab_size": int(len(search.best_estimator_.named_steps["vec"].vocabulary_)),
        }
        fitted[name] = search.best_estimator_
        print(f"\n=== {name} ===  best params {search.best_params_}  CV F1 {search.best_score_:.4f}")
        print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))

    best_name = max(results, key=lambda n: results[n]["cv_f1"])
    best = fitted[best_name]
    print(f"Selected: {best_name}")

    # ---- Confusion matrices ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, model) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["ham", "spam"],
                                              cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{name}\nprecision {results[name]['test_spam_precision']:.3f} · recall {results[name]['test_spam_recall']:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    # ---- Which words scream "spam"? ----------------------------------------
    vec, nb = best.named_steps["vec"], best.named_steps["nb"]
    vocab = np.array(vec.get_feature_names_out())
    log_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    order = np.argsort(log_ratio)
    spammy = pd.Series(log_ratio[order[-15:]], index=vocab[order[-15:]])
    hammy = pd.Series(log_ratio[order[:15]], index=vocab[order[:15]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    spammy.plot.barh(ax=axes[0], color="#C44E52")
    axes[0].set_title("Most 'spammy' tokens")
    hammy.iloc[::-1].plot.barh(ax=axes[1], color="#4C72B0")
    axes[1].set_title("Most 'hammy' tokens")
    for ax in axes:
        ax.set_xlabel("log P(token | spam) − log P(token | ham)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "indicative_tokens.png", dpi=120)
    plt.close(fig)

    # ---- Look at the mistakes — this is where you actually learn things ----
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    errors = pd.DataFrame({"text": raw_test.values, "actual": y_test.values, "predicted": pred, "p_spam": proba.round(3)})
    errors = errors[errors["actual"] != errors["predicted"]]
    errors["actual"] = errors["actual"].map({0: "ham", 1: "spam"})
    errors["predicted"] = errors["predicted"].map({0: "ham", 1: "spam"})
    errors.to_csv(OUTPUTS_DIR / "misclassified.csv", index=False)

    joblib.dump(best, MODELS_DIR / "spam_model.joblib")
    summary = {
        "selected_model": best_name,
        "n_messages": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "spam_share": round(float(df["is_spam"].mean()), 4),
        "models": results,
        "top_spam_tokens": list(spammy.index[::-1][:10]),
        "n_misclassified": int(len(errors)),
        "example_errors": errors.head(6).to_dict(orient="records"),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(errors)} misclassified messages saved to outputs/misclassified.csv")
    print(f"Model saved to {MODELS_DIR / 'spam_model.joblib'}")


if __name__ == "__main__":
    main()
