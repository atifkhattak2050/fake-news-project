import os, json, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EncodedDS(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self): return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

def chunk_encode(texts, labels, tok, max_len, stride):
    # Expand each document into multiple overlapping chunks.
    all_ids = []
    all_mask = []
    all_labels = []

    for t, y in zip(texts, labels):
        enc = tok(
            t,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt"
        )
        # enc["input_ids"] has shape (num_chunks, max_len)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        n = ids.shape[0]
        all_ids.append(ids)
        all_mask.append(mask)
        all_labels.append(torch.full((n,), y, dtype=torch.long))

    input_ids = torch.cat(all_ids, dim=0)
    attention_mask = torch.cat(all_mask, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    return input_ids, attention_mask, labels_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=os.environ.get("RUN_ID", "20251214-adapt_chunks"))
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mode", choices=["btt_only", "ax_plus_btt"], default="ax_plus_btt")
    ap.add_argument("--btt_dev_ratio", type=float, default=0.2)
    ap.add_argument("--btt_upsample", type=int, default=2)  # chunking already expands data; keep modest

    args = ap.parse_args()
    set_seed(args.seed)

    btt_path = Path("data_intermediate/btt_final.parquet")
    assert btt_path.exists(), f"Missing {btt_path}"
    btt = pd.read_parquet(btt_path)
    btt_train_full = btt[btt["split"] == "train"].copy()

    btt_tr, btt_dev = train_test_split(
        btt_train_full,
        test_size=args.btt_dev_ratio,
        random_state=args.seed,
        stratify=btt_train_full["label"]
    )

    # Build doc-level training pool
    train_texts = btt_tr["text_clean"].astype(str).tolist()
    train_labels = [LABEL2ID[x] for x in btt_tr["label"].astype(str).tolist()]

    # Optional: add AX train docs (transfer)
    if args.mode == "ax_plus_btt":
        ax_path = Path("data_intermediate/ax_splits.parquet")
        assert ax_path.exists(), f"Missing {ax_path}"
        ax = pd.read_parquet(ax_path)
        ax_train = ax[ax["split"] == "train"].copy()
        train_texts += ax_train["text_clean"].astype(str).tolist()
        train_labels += [LABEL2ID[x] for x in ax_train["label"].astype(str).tolist()]

    # Upsample BTT docs a bit (helps correct your real→fake bias on target style)
    train_texts = train_texts + (btt_tr["text_clean"].astype(str).tolist() * max(0, args.btt_upsample - 1))
    train_labels = train_labels + ([LABEL2ID[x] for x in btt_tr["label"].astype(str).tolist()] * max(0, args.btt_upsample - 1))

    dev_texts = btt_dev["text_clean"].astype(str).tolist()
    dev_labels = [LABEL2ID[x] for x in btt_dev["label"].astype(str).tolist()]

    # Tokenizer load (ignore weird warning sources gracefully)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Chunk expansion
    tr_ids, tr_mask, tr_y = chunk_encode(train_texts, train_labels, tok, args.max_len, args.stride)
    dv_ids, dv_mask, dv_y = chunk_encode(dev_texts, dev_labels, tok, args.max_len, args.stride)

    ds_train = EncodedDS(tr_ids, tr_mask, tr_y)
    ds_dev   = EncodedDS(dv_ids, dv_mask, dv_y)

    out_dir = Path(f"runs/{args.run_id}/xlmr_{args.mode}_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    ta = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    metrics = trainer.evaluate()

    (out_dir / "btt_dev_chunk_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model(str(out_dir / "model"))
    tok.save_pretrained(str(out_dir / "model"))

    print("SAVED:", out_dir / "model")
    print("BTT dev (chunk-level) macro-F1:", metrics.get("eval_macro_f1"))
    print("Train chunks:", len(ds_train), "Dev chunks:", len(ds_dev))

if __name__ == "__main__":
    main()