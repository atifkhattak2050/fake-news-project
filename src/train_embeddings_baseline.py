# src/train_embeddings_baseline.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

RUN_ID = "20251213-ax_baseline_setup"
SEED = 42

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AX_SPLITS = Path("data_intermediate/ax_splits.parquet")
OUT_DIR = Path(f"runs/{RUN_ID}/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path(f"runs/{RUN_ID}/embedding_results_ax.json")

def get_split(df, split):
    d = df[df["split"] == split].copy()
    X = d["text_clean"].astype(str).tolist()
    y = d["label"].astype(str).tolist()
    return X, y

def eval_model(clf, X_emb, y_true):
    pred = clf.predict(X_emb)
    rep = classification_report(y_true, pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, pred, labels=["real", "fake"])
    return rep, cm.tolist()

def main():
    df = pd.read_parquet(AX_SPLITS)

    X_train, y_train = get_split(df, "train")
    X_val, y_val = get_split(df, "val")
    X_test, y_test = get_split(df, "test")

    print("Loading embedding model:", MODEL_NAME)
    emb_model = SentenceTransformer(MODEL_NAME, device="cpu")

    # Encode
    # normalize_embeddings=True helps cosine geometry and often improves LR stability
    print("Encoding train...")
    E_train = emb_model.encode(X_train, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding val...")
    E_val = emb_model.encode(X_val, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding test...")
    E_test = emb_model.encode(X_test, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    E_train = np.asarray(E_train)
    E_val = np.asarray(E_val)
    E_test = np.asarray(E_test)

    # Classifier
    clf = LogisticRegression(
        max_iter=4000,
        random_state=SEED,
        class_weight="balanced"
    )
    clf.fit(E_train, y_train)

    rep_val, cm_val = eval_model(clf, E_val, y_val)
    rep_test, cm_test = eval_model(clf, E_test, y_test)

    # Save artifacts
    joblib.dump({"model_name": MODEL_NAME}, OUT_DIR / "sentence_embedder_meta.joblib")
    joblib.dump(clf, OUT_DIR / "sentence_embeddings_logreg.joblib")

    results = {
        "model_name": MODEL_NAME,
        "classifier": {"type": "LogisticRegression", "class_weight": "balanced", "max_iter": 4000},
        "val": {"report": rep_val, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_val},
        "test": {"report": rep_test, "confusion_matrix_labels": ["real","fake"], "confusion_matrix": cm_test},
        "headline": {
            "macro_f1_val": rep_val["macro avg"]["f1-score"],
            "macro_f1_test": rep_test["macro avg"]["f1-score"]
        }
    }

    REPORT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", REPORT_PATH)
    print(f"sentence_embeddings_logreg: macroF1 val={results['headline']['macro_f1_val']:.4f} test={results['headline']['macro_f1_test']:.4f}")

if __name__ == "__main__":
    main()
