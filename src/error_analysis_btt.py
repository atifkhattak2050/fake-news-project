# src/error_analysis_btt.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

RUN_ID = "20251213-ax_baseline_setup"

BTT_IN = Path("data_intermediate/btt_final.parquet")
MODELS_DIR = Path(f"runs/{RUN_ID}/models")
OUT_JSON = Path(f"runs/{RUN_ID}/error_analysis_btt.json")

LABELS = ["real", "fake"]

def main():
    btt = pd.read_parquet(BTT_IN)

    # evaluate on test split if present
    if "split" in btt.columns and (btt["split"] == "test").any():
        btt = btt[btt["split"] == "test"].copy()

    X = btt["text_clean"].astype(str).tolist()
    y_true = btt["label"].astype(str).tolist()

    results = {}

    # -------- Char TF-IDF + LinearSVC (primary) --------
    model = joblib.load(MODELS_DIR / "char_tfidf_linearsvc.joblib")
    y_pred = model.predict(X)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)

    results["char_tfidf_linearsvc"] = {
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "report": report
    }

    # Save
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", OUT_JSON)

    # Print human-readable summary
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("          pred_real   pred_fake")
    print(f"true_real    {cm[0][0]:5d}        {cm[0][1]:5d}")
    print(f"true_fake    {cm[1][0]:5d}        {cm[1][1]:5d}")

    print("\nKey rates:")
    real_recall = report["real"]["recall"]
    fake_recall = report["fake"]["recall"]
    print(f"Recall(real) = {real_recall:.3f}")
    print(f"Recall(fake) = {fake_recall:.3f}")

if __name__ == "__main__":
    main()
