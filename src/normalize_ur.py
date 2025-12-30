# src/normalize_ur.py
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

RUN_ID = "20251213-ax_baseline_setup"

AX_IN = Path("data_intermediate/ax_raw.parquet")
BTT_IN = Path("data_intermediate/btt_raw.parquet")

AX_OUT = Path("data_intermediate/ax_clean.parquet")
BTT_OUT = Path("data_intermediate/btt_clean.parquet")

REPORT = Path(f"runs/{RUN_ID}/normalization_report.json")

# Common zero-width and non-printing characters seen in Urdu corpora
ZW_CHARS = [
    "\u200b",  # zero width space
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\ufeff",  # BOM
]

# Minimal, conservative Urdu canonicalization map
# (kept intentionally small to avoid semantic distortion)
URDU_CHAR_MAP = {
    "ي": "ی",   # Arabic Yeh -> Urdu Yeh
    "ى": "ی",   # Alef Maksura -> Urdu Yeh
    "ك": "ک",   # Arabic Kaf -> Urdu Kaf
    "ۃ": "ہ",   # do-chashmi heh-like normalization (conservative choice)
}

ZW_RE = re.compile("|".join(map(re.escape, ZW_CHARS)))

def normalize_urdu(text: str) -> str:
    if text is None:
        return ""

    # Unicode normalization
    t = unicodedata.normalize("NFKC", text)

    # Remove zero-width/non-printing
    t = ZW_RE.sub("", t)

    # Character canonicalization
    for src, dst in URDU_CHAR_MAP.items():
        t = t.replace(src, dst)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t

def run_one(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict]:
    before = df["text_raw"].astype(str)
    after = before.apply(normalize_urdu)

    changed = (before != after)
    pct_changed = float(changed.mean() * 100.0)

    out = df.copy()
    out["text_clean"] = after
    out["clean_len"] = out["text_clean"].str.len()

    # Drop empties (should be extremely rare)
    before_n = len(out)
    out = out[out["clean_len"] > 0].copy()
    dropped_empty = before_n - len(out)

    examples = []
    # store up to 50 examples for auditability
    for idx in out.index[changed.loc[out.index]].tolist()[:50]:
        examples.append({
            "before": df.loc[idx, "text_raw"][:300],
            "after": out.loc[idx, "text_clean"][:300],
        })

    rep = {
        "dataset": dataset_name,
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "pct_modified": pct_changed,
        "dropped_empty_after_cleaning": int(dropped_empty),
        "unique_chars_before": int(len(set("".join(before.head(2000).tolist())))),
        "unique_chars_after": int(len(set("".join(after.head(2000).tolist())))),
        "len_stats_before": {
            "min": int(before.str.len().min()),
            "median": float(before.str.len().median()),
            "p95": float(before.str.len().quantile(0.95)),
            "max": int(before.str.len().max()),
        },
        "len_stats_after": {
            "min": int(out["clean_len"].min()),
            "median": float(out["clean_len"].median()),
            "p95": float(out["clean_len"].quantile(0.95)),
            "max": int(out["clean_len"].max()),
        },
        "examples_before_after": examples
    }
    return out, rep

def main():
    assert AX_IN.exists(), f"Missing {AX_IN}"
    assert BTT_IN.exists(), f"Missing {BTT_IN}"

    ax = pd.read_parquet(AX_IN)
    btt = pd.read_parquet(BTT_IN)

    ax_out, ax_rep = run_one(ax, "ax_to_grind")
    btt_out, btt_rep = run_one(btt, "bend_the_truth")

    AX_OUT.parent.mkdir(parents=True, exist_ok=True)
    ax_out.to_parquet(AX_OUT, index=False)
    btt_out.to_parquet(BTT_OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"ax": ax_rep, "btt": btt_rep}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE:", AX_OUT)
    print("WROTE:", BTT_OUT)
    print("WROTE:", REPORT)
    print(json.dumps({"ax_pct_modified": ax_rep["pct_modified"], "btt_pct_modified": btt_rep["pct_modified"]}, indent=2))

if __name__ == "__main__":
    main()
