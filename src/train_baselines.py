# src/train_baselines.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

RUN_ID = "20251213-ax_baseline_setup"
SEED = 42

AX_SPLITS = Path("data_intermediate/ax_splits.parquet")

OUT_DIR = Path(f"runs/{RUN_ID}/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path(f"runs/{RUN_ID}/baseline_results_ax.json")

def load_split(df, split):
    d = df[df["split"] == split].copy()
    X = d["text_clean"].astype(str).tolist()
    y = d["label"].astype(str).tolist()
    return X, y

def eval_model(model, X, y):
    pred = model.predict(X)
    rep = classification_report(y, pred, output_dict=True, digits=4)
    cm = confusion_matrix(y, pred, labels=["real", "fake"])
    return rep, cm.tolist()

def main():
    df = pd.read_parquet(AX_SPLITS)
    assert "text_clean" in df.columns and "label" in df.columns and "split" in df.columns

    X_train, y_train = load_split(df, "train")
    X_val, y_val = load_split(df, "val")
    X_test, y_test = load_split(df, "test")

    experiments = {}

    # -------------------------
    # Baseline 1: Char TF-IDF + LinearSVC
    # -------------------------
    char_svm = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LinearSVC())
    ])

    char_svm.fit(X_train, y_train)

    rep_val, cm_val = eval_model(char_svm, X_val, y_val)
    rep_test, cm_test = eval_model(char_svm, X_test, y_test)

    joblib.dump(char_svm, OUT_DIR / "char_tfidf_linearsvc.joblib")

    experiments["char_tfidf_linearsvc"] = {
        "val": {"report": rep_val, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_val},
        "test": {"report": rep_test, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_test},
        "vectorizer": {"analyzer": "char", "ngram_range": [3,5], "min_df": 2, "max_df": 0.95},
        "model": {"type": "LinearSVC"}
    }

    # -------------------------
    # Baseline 2: Word TF-IDF + LogisticRegression
    # -------------------------
    word_lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=1,
            random_state=SEED
        ))
    ])

    word_lr.fit(X_train, y_train)

    rep_val, cm_val = eval_model(word_lr, X_val, y_val)
    rep_test, cm_test = eval_model(word_lr, X_test, y_test)

    joblib.dump(word_lr, OUT_DIR / "word_tfidf_logreg.joblib")

    experiments["word_tfidf_logreg"] = {
        "val": {"report": rep_val, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_val},
        "test": {"report": rep_test, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_test},
        "vectorizer": {"analyzer": "word", "ngram_range": [1,2], "min_df": 2, "max_df": 0.95},
        "model": {"type": "LogisticRegression", "max_iter": 2000}
    }

    REPORT_PATH.write_text(json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", REPORT_PATH)
    print("WROTE MODELS TO:", OUT_DIR)

    # Print quick headline metrics
    for k, v in experiments.items():
        mf1_val = v["val"]["report"]["macro avg"]["f1-score"]
        mf1_test = v["test"]["report"]["macro avg"]["f1-score"]
        print(f"{k}: macroF1 val={mf1_val:.4f} test={mf1_test:.4f}")

if __name__ == "__main__":
    main()
