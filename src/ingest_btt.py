# src/ingest_btt.py
import json
import hashlib
import re
from pathlib import Path

import pandas as pd

RUN_ID = "20251213-ax_baseline_setup"
BTT_ROOT = Path("data_raw/bend_the_truth")
OUT_PARQUET = Path("data_intermediate/btt_raw.parquet")
SUMMARY_JSON = Path(f"runs/{RUN_ID}/data_summary_btt.json")

def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def read_txt(path: Path) -> str:
    # Urdu text files are typically UTF-8; handle fallback safely.
    for enc in ("utf-8", "utf-8-sig", "cp1256"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            pass
    return path.read_text(errors="ignore")

def infer_domain(filename: str) -> str | None:
    # README says category is encoded in filenames (e.g., sp, tech, etc.)
    # We infer a leading alphabetic prefix.
    m = re.match(r"^([A-Za-z]+)[_\-].*", filename)
    if m:
        return m.group(1).lower()
    # Sometimes filenames start like "sp1.txt" or "tech23.txt"
    m = re.match(r"^([A-Za-z]+)\d+.*", filename)
    if m:
        return m.group(1).lower()
    return None

def main():
    assert BTT_ROOT.exists(), f"Missing Bend-the-Truth root: {BTT_ROOT}"

    rows = []
    for split in ("Train", "Test"):
        split_c = split.lower()  # canonical: train/test
        for label_folder in ("Real", "Fake"):
            label_c = label_folder.lower()  # canonical: real/fake

            base = BTT_ROOT / split / label_folder
            assert base.exists(), f"Missing folder: {base}"

            for fp in sorted(base.glob("*.txt")):
                text = read_txt(fp).strip()
                if len(text) < 30:
                    continue

                rows.append({
                    "id": stable_id(text),
                    "dataset": "bend_the_truth",
                    "split": split_c,
                    "label": label_c,
                    "text_raw": text,
                    "filename": fp.name,
                    "domain": infer_domain(fp.name),
                    "text_len": len(text),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No .txt files ingested; check folder paths and extensions.")

    # Sanity: both classes exist in each split (or at least globally)
    if set(df["label"].unique()) != {"real", "fake"}:
        raise ValueError(f"Unexpected labels: {df['label'].unique()}")
    if set(df["split"].unique()) != {"train", "test"}:
        raise ValueError(f"Unexpected splits: {df['split'].unique()}")

    # Save
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # Summary
    summary = {
        "rows_total_ingested": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "label_distribution_overall": df["label"].value_counts().to_dict(),
        "label_distribution_by_split": (
            df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
        ),
        "domain_coverage_top10": df["domain"].value_counts().head(10).to_dict(),
        "text_len_stats": {
            "min": int(df["text_len"].min()),
            "median": float(df["text_len"].median()),
            "p95": float(df["text_len"].quantile(0.95)),
            "max": int(df["text_len"].max()),
        },
        "columns_written": list(df.columns),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", OUT_PARQUET)
    print("WROTE:", SUMMARY_JSON)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
