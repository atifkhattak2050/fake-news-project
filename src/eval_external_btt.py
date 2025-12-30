# src/eval_external_btt.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

RUN_ID = "20251213-ax_baseline_setup"

BTT_IN = Path("data_intermediate/btt_final.parquet")

MODELS_DIR = Path(f"runs/{RUN_ID}/models")
OUT_JSON = Path(f"runs/{RUN_ID}/external_results_btt.json")

def eval_preds(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
    return rep, cm.tolist()

def main():
    assert BTT_IN.exists(), f"Missing {BTT_IN}"
    btt = pd.read_parquet(BTT_IN)

    # Prefer evaluating on the "test" split if present; else evaluate entire BTT
    if "split" in btt.columns and (btt["split"] == "test").any():
        btt_eval = btt[btt["split"] == "test"].copy()
        scope = "btt_test_split_only"
    else:
        btt_eval = btt.copy()
        scope = "btt_all_rows"

    X = btt_eval["text_clean"].astype(str).tolist()
    y = btt_eval["label"].astype(str).tolist()

    results = {"scope": scope, "rows": int(len(btt_eval)), "models": {}}

    # 1) Char TF-IDF + LinearSVC
    char_model_path = MODELS_DIR / "char_tfidf_linearsvc.joblib"
    char_model = joblib.load(char_model_path)
    y_pred = char_model.predict(X)
    rep, cm = eval_preds(y, y_pred)
    results["models"]["char_tfidf_linearsvc"] = {
        "report": rep,
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": cm,
        "headline": {"macro_f1": rep["macro avg"]["f1-score"]}
    }

    # 2) Word TF-IDF + LogisticRegression
    word_model_path = MODELS_DIR / "word_tfidf_logreg.joblib"
    word_model = joblib.load(word_model_path)
    y_pred = word_model.predict(X)
    rep, cm = eval_preds(y, y_pred)
    results["models"]["word_tfidf_logreg"] = {
        "report": rep,
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": cm,
        "headline": {"macro_f1": rep["macro avg"]["f1-score"]}
    }

    # 3) Sentence embeddings + LogisticRegression
    emb_clf_path = MODELS_DIR / "sentence_embeddings_logreg.joblib"
    emb_meta_path = MODELS_DIR / "sentence_embedder_meta.joblib"

    emb_meta = joblib.load(emb_meta_path)
    model_name = emb_meta["model_name"]

    print("Loading embedder for external eval:", model_name)
    emb_model = SentenceTransformer(model_name, device="cpu")

    E = emb_model.encode(X, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    E = np.asarray(E)

    emb_clf = joblib.load(emb_clf_path)
    y_pred = emb_clf.predict(E)
    rep, cm = eval_preds(y, y_pred)
    results["models"]["sentence_embeddings_logreg"] = {
        "embedder": model_name,
        "report": rep,
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": cm,
        "headline": {"macro_f1": rep["macro avg"]["f1-score"]}
    }

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", OUT_JSON)

    for k, v in results["models"].items():
        print(f"{k}: BTT macroF1={v['headline']['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
