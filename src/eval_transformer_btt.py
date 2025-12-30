import os, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["real", "fake"]
LABEL2ID = {"real": 0, "fake": 1}

def window_logits(model, tok, text, max_len=256, stride=128, device="cuda"):
    # tokenize without truncation first
    enc = tok(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]

    # if short, single pass
    if input_ids.numel() <= max_len:
        batch = tok(text, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch).logits[0].detach().cpu().numpy()
        return out

    # sliding windows over tokens
    logits_list = []
    start = 0
    while start < input_ids.numel():
        end = min(start + max_len, input_ids.numel())
        ids = input_ids[start:end]
        am  = attn[start:end]
        # pad
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), tok.pad_token_id, dtype=torch.long)])
            am  = torch.cat([am,  torch.zeros((pad_len,), dtype=torch.long)])
        batch = {"input_ids": ids.unsqueeze(0).to(device), "attention_mask": am.unsqueeze(0).to(device)}
        with torch.no_grad():
            logit = model(**batch).logits[0].detach().cpu().numpy()
        logits_list.append(logit)
        if end == input_ids.numel():
            break
        start += stride

    # aggregate logits (mean works well)
    return np.mean(np.stack(logits_list, axis=0), axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=os.environ.get("RUN_ID", "20251214-xlmr"))
    ap.add_argument("--model_dir", required=True)  # path to saved model folder
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    btt_path = Path("data_intermediate/btt_final.parquet")
    assert btt_path.exists(), f"Missing {btt_path}"
    btt = pd.read_parquet(btt_path)
    btt_test = btt[btt["split"] == "test"].copy()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    y_true = btt_test["label"].astype(str).tolist()
    texts  = btt_test["text_clean"].astype(str).tolist()

    preds = []
    for t in texts:
        lg = window_logits(model, tok, t, max_len=args.max_len, stride=args.stride, device=device)
        preds.append(int(np.argmax(lg)))

    y_pred = [LABELS[i] for i in preds]

    macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)

    out = {
        "btt_macro_f1": float(macro),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm,
        "report": rep,
        "max_len": args.max_len,
        "stride": args.stride,
        "model_dir": args.model_dir
    }

    out_path = Path(f"runs/{args.run_id}/btt_eval_transformer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BTT macro-F1:", macro)
    print("WROTE:", out_path)

if __name__ == "__main__":
    main()
