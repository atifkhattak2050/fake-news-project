"""
LLM zero-shot baseline: classify BTT Test articles as REAL or FAKE using Claude via CLI.

Usage:
    python src/llm_baseline_btt.py

Outputs:
    runs/<RUN_ID>/llm_btt_test_predictions.csv   — file, true, pred, raw_output
    runs/<RUN_ID>/llm_btt_test_results.json       — metrics (same format as paper)
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "20251214-ax_plus_btt_chunks_v2"
BTT_ROOT = Path("data_raw/bend_the_truth")
OUT_DIR = Path(f"runs/{RUN_ID}")
PRED_CSV = OUT_DIR / "llm_btt_test_predictions.csv"
RESULT_JSON = OUT_DIR / "llm_btt_test_results.json"

MIN_TEXT_LEN = 30
MAX_RETRIES = 3
RETRY_DELAY = 5        # seconds between retries
INTER_CALL_DELAY = 1   # seconds between successful calls

SYSTEM_PROMPT = (
    "You are a fake-news detection classifier for Urdu news articles. "
    "Read the article provided by the user and decide whether it is a REAL "
    "(genuine / authentic) news article or a FAKE (fabricated / false) news article. "
    "Output ONLY the single word REAL or FAKE. Do not output any other text."
)

USER_PROMPT_TEMPLATE = (
    "Classify the following Urdu news article as REAL or FAKE.\n\n"
    "---BEGIN ARTICLE---\n{article}\n---END ARTICLE---"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def read_txt(path: Path) -> str:
    """Read a text file with encoding fallback (UTF-8 -> UTF-8-SIG -> CP1256)."""
    for enc in ("utf-8", "utf-8-sig", "cp1256"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            pass
    return path.read_text(errors="ignore")


def collect_btt_test_files():
    """Return list of (filepath, true_label) for BTT Test split."""
    items = []
    for label_folder, label in [("Real", "real"), ("Fake", "fake")]:
        folder = BTT_ROOT / "Test" / label_folder
        assert folder.exists(), f"Missing: {folder}"
        for fp in sorted(folder.glob("*.txt")):
            text = read_txt(fp).strip()
            if len(text) < MIN_TEXT_LEN:
                continue
            items.append((fp, label, text))
    return items


def classify_one(text: str, retries: int = MAX_RETRIES) -> tuple[str, str]:
    """
    Call claude -p to classify a single article.
    Returns (prediction, raw_output).
    prediction is 'real', 'fake', or 'INVALID'.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(article=text)

    for attempt in range(1, retries + 1):
        try:
            # On Windows, use .cmd wrapper and pipe prompt via stdin
            # to avoid encoding issues with Urdu text in CLI arguments
            claude_cmd = r"C:\Users\AHT\AppData\Roaming\npm\claude.cmd"
            result = subprocess.run(
                [
                    claude_cmd, "-p",
                    "--system-prompt", SYSTEM_PROMPT,
                    "--model", "sonnet",
                ],
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="replace",
            )
            raw = result.stdout.strip()

            if result.returncode != 0:
                err = result.stderr.strip()
                print(f"    [attempt {attempt}] claude error (rc={result.returncode}): {err}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY)
                    continue
                return ("INVALID", f"ERROR: {err}")

            # Parse output
            token = raw.upper().strip().rstrip(".")
            if token == "REAL":
                return ("real", raw)
            elif token == "FAKE":
                return ("fake", raw)
            else:
                # Try to find REAL or FAKE anywhere in the response
                if re.search(r"\bFAKE\b", raw, re.IGNORECASE) and not re.search(r"\bREAL\b", raw, re.IGNORECASE):
                    return ("fake", raw)
                elif re.search(r"\bREAL\b", raw, re.IGNORECASE) and not re.search(r"\bFAKE\b", raw, re.IGNORECASE):
                    return ("real", raw)
                else:
                    return ("INVALID", raw)

        except subprocess.TimeoutExpired:
            print(f"    [attempt {attempt}] timeout")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue
            return ("INVALID", "TIMEOUT")

        except Exception as e:
            print(f"    [attempt {attempt}] exception: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue
            return ("INVALID", f"EXCEPTION: {e}")

    return ("INVALID", "MAX_RETRIES_EXCEEDED")


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute accuracy, macro-F1, per-class P/R/F1, confusion matrix."""
    labels = ["real", "fake"]
    n = len(y_true)
    assert n == len(y_pred)

    # Confusion matrix: rows = true, cols = pred
    cm = {t: {p: 0 for p in labels + ["INVALID"]} for t in labels}
    for t, p in zip(y_true, y_pred):
        if p in cm[t]:
            cm[t][p] += 1
        else:
            cm[t]["INVALID"] += 1

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    report = {}
    f1s = []
    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in labels if other != lbl)
        fn = sum(cm[lbl][other] for other in labels + ["INVALID"] if other != lbl)
        support = sum(cm[lbl].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

        report[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": float(support),
        }

    macro_f1 = sum(f1s) / len(f1s)

    # Standard 2x2 confusion matrix (real, fake order)
    cm_matrix = [
        [cm["real"]["real"], cm["real"]["fake"]],
        [cm["fake"]["real"], cm["fake"]["fake"]],
    ]

    invalid_count = sum(cm[t]["INVALID"] for t in labels)

    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": sum(report[l]["precision"] for l in labels) / len(labels),
        "recall": sum(report[l]["recall"] for l in labels) / len(labels),
        "f1-score": macro_f1,
        "support": float(n),
    }

    return {
        "report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm_matrix,
        "headline": {"macro_f1": macro_f1, "accuracy": accuracy},
        "invalid_count": invalid_count,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Record metadata
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_name = "Claude Sonnet 4.5 (via claude -p --model sonnet)"

    print(f"Run date : {run_date}")
    print(f"Model    : {model_name}")
    print(f"System   : {SYSTEM_PROMPT[:80]}...")
    print()

    # Collect test files
    items = collect_btt_test_files()
    print(f"BTT Test files collected: {len(items)}")
    print()

    # Check for existing partial results (resume support)
    done = {}
    if PRED_CSV.exists():
        with open(PRED_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["file"]] = row
        print(f"Resuming: {len(done)} predictions already done, {len(items) - len(done)} remaining")
        print()

    # Open CSV in append mode
    write_header = not PRED_CSV.exists() or len(done) == 0
    csvfile = open(PRED_CSV, "a" if not write_header else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=["file", "true", "pred", "raw_output"])
    if write_header:
        writer.writeheader()

    y_true_all = []
    y_pred_all = []
    total = len(items)

    for i, (fp, true_label, text) in enumerate(items):
        fname = fp.name

        if fname in done:
            y_true_all.append(done[fname]["true"])
            y_pred_all.append(done[fname]["pred"])
            continue

        print(f"[{i+1}/{total}] {fname} (true={true_label}) ...", end=" ", flush=True)
        pred, raw = classify_one(text)
        mark = "OK" if pred == true_label else "WRONG"
        print(f"pred={pred} [{mark}]")

        writer.writerow({
            "file": fname,
            "true": true_label,
            "pred": pred,
            "raw_output": raw,
        })
        csvfile.flush()

        y_true_all.append(true_label)
        y_pred_all.append(pred)

        if i < total - 1:
            time.sleep(INTER_CALL_DELAY)

    csvfile.close()
    print(f"\nPredictions saved to {PRED_CSV}")

    # Compute metrics
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics["meta"] = {
        "run_date": run_date,
        "model": model_name,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "n_files": total,
        "invalid_count": metrics["invalid_count"],
    }

    RESULT_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {RESULT_JSON}")

    # Print summary
    print("\n" + "=" * 60)
    print("LLM BASELINE RESULTS (BTT Test)")
    print("=" * 60)
    r = metrics["report"]
    print(f"Accuracy  : {r['accuracy']:.4f}")
    print(f"Macro-F1  : {r['macro avg']['f1-score']:.4f}")
    print()
    print(f"{'':8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
    for lbl in ["real", "fake"]:
        print(f"{lbl:8s} {r[lbl]['precision']:8.4f} {r[lbl]['recall']:8.4f} "
              f"{r[lbl]['f1-score']:8.4f} {r[lbl]['support']:8.0f}")
    print()
    cm = metrics["confusion_matrix"]
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"         {'real':>8s} {'fake':>8s}")
    for i, lbl in enumerate(["real", "fake"]):
        print(f"{lbl:8s} {cm[i][0]:8d} {cm[i][1]:8d}")
    if metrics["invalid_count"] > 0:
        print(f"\nINVALID responses: {metrics['invalid_count']}")


if __name__ == "__main__":
    main()
