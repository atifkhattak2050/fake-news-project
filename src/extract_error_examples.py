# src/extract_error_examples.py
from pathlib import Path
import pandas as pd
import joblib

RUN_ID = "20251213-ax_baseline_setup"

BTT_IN = Path("data_intermediate/btt_final.parquet")
MODEL_PATH = Path(f"runs/{RUN_ID}/models/char_tfidf_linearsvc.joblib")
OUT_CSV = Path(f"runs/{RUN_ID}/btt_error_examples.csv")

def main():
    df = pd.read_parquet(BTT_IN)

    if "split" in df.columns and (df["split"] == "test").any():
        df = df[df["split"] == "test"].copy()

    model = joblib.load(MODEL_PATH)

    df["pred"] = model.predict(df["text_clean"].astype(str))

    errors = df[df["label"] != df["pred"]].copy()

    # Label error type
    errors["error_type"] = errors["label"] + "_as_" + errors["pred"]

    # Sort by text length (shorter ones are easier to show in paper)
    errors = errors.sort_values("text_len")

    errors[[
        "label",
        "pred",
        "error_type",
        "text_clean",
        "domain"
    ]].head(20).to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("WROTE:", OUT_CSV)
    print("Error type counts:")
    print(errors["error_type"].value_counts())

if __name__ == "__main__":
    main()
