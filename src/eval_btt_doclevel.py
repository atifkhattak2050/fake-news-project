import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def make_windows(token_ids, max_len, stride, tok):
    # ensure final sequence length <= max_len after adding special tokens
    n_special = tok.num_special_tokens_to_add(pair=False)
    content_len = max_len - n_special
    if content_len <= 8:
        raise ValueError(f"max_len too small after specials: {max_len} -> content_len={content_len}")

    step = max(1, content_len - stride)
    windows = []
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start+content_len]
        if not chunk:
            break
        input_ids = tok.build_inputs_with_special_tokens(chunk)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]  # safety
        attn = [1] * len(input_ids)
        windows.append((input_ids, attn))
        if start + content_len >= len(token_ids):
            break
    return windows

@torch.no_grad()
def chunk_probs(model, tok, text, device, max_len, stride, batch_size=16):
    # tokenize once without specials, then window on token ids
    ids = tok(text, add_special_tokens=False)["input_ids"]
    wins = make_windows(ids, max_len=max_len, stride=stride, tok=tok)
    if not wins:
        return [0.0]

    probs = []
    # mini-batch windows
    for i in range(0, len(wins), batch_size):
        batch = wins[i:i+batch_size]
        maxL = max(len(x[0]) for x in batch)
        input_ids = []
        attn = []
        for ids_i, attn_i in batch:
            pad = maxL - len(ids_i)
            input_ids.append(ids_i + [tok.pad_token_id] * pad)
            attn.append(attn_i + [0] * pad)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn = torch.tensor(attn, dtype=torch.long, device=device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs.extend(p.tolist())
    return probs  # list of [p0, p1] per window

def aggregate(p_fake_list, agg="mean", k=3):
    if agg == "mean":
        return float(np.mean(p_fake_list))
    if agg == "max":
        return float(np.max(p_fake_list))
    if agg == "topk":
        k = min(k, len(p_fake_list))
        return float(np.mean(sorted(p_fake_list, reverse=True)[:k]))
    raise ValueError(f"Unknown agg: {agg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--agg", choices=["mean","max","topk"], default="mean")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--run_id", default=None)  # if set, write outputs into runs/<run_id>/
    args = ap.parse_args()

    btt = pd.read_parquet("data_intermediate/btt_final.parquet")
    df = btt[btt["split"] == args.split].copy()
    # labels are already canonical in your pipeline: real/fake
    y_true = df["label"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tok.model_max_length = 10**9
    tok.init_kwargs["model_max_length"] = 10**9
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # robust label mapping
    label2id = model.config.label2id or {"real": 0, "fake": 1}
    if "fake" in label2id:
        fake_id = int(label2id["fake"])
    else:
        fake_id = 1  # fallback

    # compute doc scores
    doc_pfake = []
    for t in df["text_clean"].tolist():
        probs = chunk_probs(model, tok, t, device, max_len=args.max_len, stride=args.stride)
        pf = [p[fake_id] for p in probs]
        doc_pfake.append(aggregate(pf, agg=args.agg, k=args.topk))

    def eval_at(th):
        y_pred = ["fake" if p >= th else "real" for p in doc_pfake]
        macro = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=["real","fake"])
        rep = classification_report(y_true, y_pred, labels=["real","fake"], output_dict=True, zero_division=0)
        return macro, cm, rep

    best = None
    if args.sweep:
        # Sweep thresholds on this split
        for th in np.arange(0.05, 0.96, 0.05):
            macro, cm, rep = eval_at(float(th))
            if (best is None) or (macro > best["macro_f1"]):
                best = {"threshold": float(th), "macro_f1": float(macro), "cm": cm.tolist(), "rep": rep}
        print("BEST:", json.dumps({"threshold": best["threshold"], "macro_f1": best["macro_f1"], "cm": best["cm"]}, ensure_ascii=False))
    else:
        macro, cm, rep = eval_at(float(args.threshold))
        best = {"threshold": float(args.threshold), "macro_f1": float(macro), "cm": cm.tolist(), "rep": rep}

    out = {
        "split": args.split,
        "max_len": args.max_len,
        "stride": args.stride,
        "agg": args.agg,
        "topk": args.topk,
        "threshold": best["threshold"],
        "macro_f1": best["macro_f1"],
        "confusion_matrix_labels": ["real","fake"],
        "confusion_matrix": best["cm"],
        "report": best["rep"],
    }

    # write result file (do NOT overwrite across settings)
    out_dir = Path("runs") / args.run_id if args.run_id else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"btt_doclevel_{args.split}_{args.agg}_L{args.max_len}_S{args.stride}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", str(out_path))
    print(f"BTT ({args.split}) macro-F1:", out["macro_f1"])
    print("CM:", out["confusion_matrix"])

if __name__ == "__main__":
    main()
