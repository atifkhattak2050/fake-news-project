# src/make_splits_ax.py
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RUN_ID = "20251213-ax_baseline_setup"
SEED = 42

AX_IN = Path("data_intermediate/ax_final.parquet")
AX_OUT = Path("data_intermediate/ax_splits.parquet")
REPORT = Path(f"runs/{RUN_ID}/split_report_ax.json")

def main():
    assert AX_IN.exists(), f"Missing {AX_IN}"
    df = pd.read_parquet(AX_IN)

    required = {"id", "label", "text_clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # First split: train vs temp (20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["label"]
    )

    # Second split: val vs test (each 10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # Sanity checks
    if set(out["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Split labels incorrect.")
    if out["id"].duplicated().any():
        raise ValueError("Duplicate IDs present after split assignment.")

    # Save
    AX_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AX_OUT, index=False)

    # Report (robust, deterministic)
    split_sizes = out.groupby("split").size().to_dict()
    label_by_split = out.groupby(["split", "label"]).size().unstack(fill_value=0)

    report = {
        "rows_total": int(len(out)),
        "split_sizes": {k: int(v) for k, v in split_sizes.items()},
        "label_distribution_overall": out["label"].value_counts().to_dict(),
        "label_by_split": label_by_split.to_dict(),
        "seed": SEED,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", AX_OUT)
    print("WROTE:", REPORT)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
