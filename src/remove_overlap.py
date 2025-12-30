# src/remove_overlap.py
import json
import hashlib
from pathlib import Path
import pandas as pd

RUN_ID = "20251213-ax_baseline_setup"

AX_IN = Path("data_intermediate/ax_dedup.parquet")
BTT_IN = Path("data_intermediate/btt_dedup.parquet")

AX_OUT = Path("data_intermediate/ax_final.parquet")
BTT_OUT = Path("data_intermediate/btt_final.parquet")

REPORT = Path(f"runs/{RUN_ID}/overlap_remediation_report.json")

def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    ax = pd.read_parquet(AX_IN)
    btt = pd.read_parquet(BTT_IN)

    # Ensure we use clean text
    for df, name in [(ax, "ax"), (btt, "btt")]:
        if "text_clean" not in df.columns:
            raise ValueError(f"{name} missing text_clean column.")

    # Compute hashes (again, to be explicit)
    ax["text_hash"] = ax["text_clean"].astype(str).apply(text_hash)
    btt["text_hash"] = btt["text_clean"].astype(str).apply(text_hash)

    # 1) Deduplicate BTT by exact text (fix btt_id_duplicates)
    btt_before = len(btt)
    btt = btt.drop_duplicates(subset=["text_hash"], keep="first").copy()
    btt_after = len(btt)

    # 2) Remove any BTT-overlapping texts from Ax
    btt_hashes = set(btt["text_hash"])
    ax_before = len(ax)
    overlap_mask = ax["text_hash"].isin(btt_hashes)
    overlap_count = int(overlap_mask.sum())

    ax = ax[~overlap_mask].copy()
    ax_after = len(ax)

    # 3) Final sanity checks
    ax_id_dups = int(ax["id"].duplicated().sum())
    btt_id_dups = int(btt["id"].duplicated().sum())
    # check overlap truly removed
    final_overlap = len(set(ax["text_hash"]).intersection(set(btt["text_hash"])))

    # Write
    ax.to_parquet(AX_OUT, index=False)
    btt.to_parquet(BTT_OUT, index=False)

    rep = {
        "btt_rows_before": int(btt_before),
        "btt_rows_after_text_dedup": int(btt_after),
        "btt_removed_exact_text_dups": int(btt_before - btt_after),
        "ax_rows_before_overlap_removal": int(ax_before),
        "ax_removed_due_to_overlap_with_btt": int(overlap_count),
        "ax_rows_after_overlap_removal": int(ax_after),
        "ax_id_duplicates_final": ax_id_dups,
        "btt_id_duplicates_final": btt_id_dups,
        "cross_corpus_exact_overlap_final": int(final_overlap),
    }
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", AX_OUT)
    print("WROTE:", BTT_OUT)
    print("WROTE:", REPORT)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
