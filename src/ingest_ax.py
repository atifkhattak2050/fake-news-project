# src/ingest_ax.py
import json
import hashlib
from pathlib import Path

import pandas as pd

RUN_ID = "20251213-ax_baseline_setup"
AX_PATH = Path("data_raw/ax_to_grind.csv")
OUT_PARQUET = Path("data_intermediate/ax_raw.parquet")
SUMMARY_JSON = Path(f"runs/{RUN_ID}/data_summary_ax.json")

SEED = 42

def stable_id(text: str) -> str:
    # Stable ID from raw text. Good for reproducibility.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return h

def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    # Try common Urdu CSV encodings
    for enc in ("utf-8", "utf-8-sig", "cp1256"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # Last resort: let pandas guess (may still fail)
    return pd.read_csv(path)

def main():
    assert AX_PATH.exists(), f"Missing: {AX_PATH}"

    df = load_csv_with_fallback(AX_PATH)

    # Explicit column binding for Ax-to-Grind
    text_col = "News Items"
    label_col = "Label"

    assert text_col in df.columns, f"Missing text column: {text_col}"
    assert label_col in df.columns, f"Missing label column: {label_col}"

        # Standardize
    out = pd.DataFrame()
    out["text_raw"] = df[text_col].astype(str)

    # Map labels -> {real,fake}
    raw_labels = df[label_col].astype(str).str.strip().str.lower()
    label_map = {
        "true": "real",
        "real": "real",
        "0": "real",
        "fake": "fake",
        "false": "fake",
        "1": "fake",
    }
    out["label"] = raw_labels.map(label_map)

    # Keep optional metadata if exists
    for col in ["domain", "Domain", "category", "Category", "year", "Year", "date", "Date", "source", "Source", "url", "URL"]:
        if col in df.columns:
            out[col.lower()] = df[col]

    out["dataset"] = "ax_to_grind"

    # Basic filtering: drop empties / very short
    out["text_len"] = out["text_raw"].str.len()
    before = len(out)
    out = out[out["text_len"] >= 30].copy()
    dropped_short = before - len(out)

    # Stable IDs
    out["id"] = out["text_raw"].apply(stable_id)

    # Reorder canonical columns
    keep_cols = ["id", "dataset", "label", "text_raw"] + [c for c in out.columns if c not in {"id","dataset","label","text_raw"}]
    out = out[keep_cols]

    # Integrity checks
    if out["label"].isna().any():
        bad = int(out["label"].isna().sum())
        raise ValueError(
            f"{bad} rows have unmapped labels. "
            f"Inspect unique raw labels in your CSV and extend label_map."
        )

    # Save
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)

    # Summary
    summary = {
        "rows_total_loaded": int(len(df)),
        "rows_after_minlen_filter": int(len(out)),
        "dropped_short_lt_30chars": int(dropped_short),
        "text_col_used": text_col,
        "label_col_used": label_col,
        "label_distribution": out["label"].value_counts().to_dict(),
        "columns_written": list(out.columns),
        "text_len_stats": {
            "min": int(out["text_len"].min()),
            "median": float(out["text_len"].median()),
            "p95": float(out["text_len"].quantile(0.95)),
            "max": int(out["text_len"].max()),
        },
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", OUT_PARQUET)
    print("WROTE:", SUMMARY_JSON)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
