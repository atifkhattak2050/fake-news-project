# src/dedup.py
import json
import hashlib
from pathlib import Path
import pandas as pd

RUN_ID = "20251213-ax_baseline_setup"

AX_IN = Path("data_intermediate/ax_clean.parquet")
BTT_IN = Path("data_intermediate/btt_clean.parquet")

AX_OUT = Path("data_intermediate/ax_dedup.parquet")
BTT_OUT = Path("data_intermediate/btt_dedup.parquet")  # BTT should remain same, but we keep consistent artifacts.

REPORT = Path(f"runs/{RUN_ID}/dedup_report.json")

def text_hash(s: str) -> str:
    # Hash of normalized text for duplicate detection
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    assert AX_IN.exists(), f"Missing {AX_IN}"
    assert BTT_IN.exists(), f"Missing {BTT_IN}"

    ax = pd.read_parquet(AX_IN)
    btt = pd.read_parquet(BTT_IN)

    # Use clean text for dedup. If text_clean not present, fail fast.
    if "text_clean" not in ax.columns:
        raise ValueError("ax_clean.parquet missing text_clean column.")
    if "text_clean" not in btt.columns:
        raise ValueError("btt_clean.parquet missing text_clean column.")

    # Hashes
    ax["text_hash"] = ax["text_clean"].astype(str).apply(text_hash)
    btt["text_hash"] = btt["text_clean"].astype(str).apply(text_hash)

    # 1) Ax internal dedup
    ax_before = len(ax)
    ax = ax.drop_duplicates(subset=["text_hash"], keep="first").copy()
    ax_after = len(ax)

    # 2) Cross-corpus overlap check (exact)
    overlap = set(ax["text_hash"]).intersection(set(btt["text_hash"]))
    overlap_count = len(overlap)

    # 3) Verify unique IDs (should be unique after dedup)
    ax_id_dups = int(ax["id"].duplicated().sum())
    btt_id_dups = int(btt["id"].duplicated().sum())

    # Persist
    AX_OUT.parent.mkdir(parents=True, exist_ok=True)
    ax.to_parquet(AX_OUT, index=False)
    btt.to_parquet(BTT_OUT, index=False)

    report = {
        "ax_rows_before": int(ax_before),
        "ax_rows_after_exact_dedup": int(ax_after),
        "ax_removed_exact_duplicates": int(ax_before - ax_after),
        "ax_id_duplicates_after": ax_id_dups,
        "btt_rows": int(len(btt)),
        "btt_id_duplicates": btt_id_dups,
        "cross_corpus_exact_overlap_count": int(overlap_count),
        # store sample overlaps for audit if any
        "cross_corpus_overlap_sample_hashes": list(sorted(list(overlap))[:20]),
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", AX_OUT)
    print("WROTE:", BTT_OUT)
    print("WROTE:", REPORT)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
