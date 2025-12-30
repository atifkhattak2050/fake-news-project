import os, json, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
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

class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=os.environ.get("RUN_ID", "20251214-xlmr"))
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_mode", choices=["ax_only", "ax_plus_btt"], default="ax_plus_btt")
    ap.add_argument("--btt_upsample", type=int, default=5)  # helps because BTT is small
    args = ap.parse_args()

    set_seed(args.seed)

    ax_path = Path("data_intermediate/ax_splits.parquet")
    btt_path = Path("data_intermediate/btt_final.parquet")
    assert ax_path.exists(), f"Missing {ax_path}"
    assert btt_path.exists(), f"Missing {btt_path}"

    ax = pd.read_parquet(ax_path)
    # AX splits: train/val/test
    ax_train = ax[ax["split"] == "train"].copy()
    ax_val   = ax[ax["split"] == "val"].copy()

    # BTT splits: train/test (already in your pipeline)
    btt = pd.read_parquet(btt_path)
    btt_train = btt[btt["split"] == "train"].copy()

    # Ensure columns
    for df, name in [(ax_train,"ax_train"), (ax_val,"ax_val"), (btt_train,"btt_train")]:
        assert "text_clean" in df.columns and "label" in df.columns, f"Missing columns in {name}"

    # Build training set
    train_texts = ax_train["text_clean"].astype(str).tolist()
    train_labels = [LABEL2ID[x] for x in ax_train["label"].astype(str).tolist()]

    if args.train_mode == "ax_plus_btt":
        bt = btt_train["text_clean"].astype(str).tolist()
        bl = [LABEL2ID[x] for x in btt_train["label"].astype(str).tolist()]
        # upsample BTT to influence decision boundary and reduce real->fake bias
        bt = bt * max(1, args.btt_upsample)
        bl = bl * max(1, args.btt_upsample)
        train_texts += bt
        train_labels += bl

    val_texts = ax_val["text_clean"].astype(str).tolist()
    val_labels = [LABEL2ID[x] for x in ax_val["label"].astype(str).tolist()]

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    ds_train = TextDS(train_texts, train_labels, tok, args.max_len)
    ds_val   = TextDS(val_texts,   val_labels,   tok, args.max_len)

    out_dir = Path(f"runs/{args.run_id}/xlmr_{args.train_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ta = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save summary
    (out_dir / "ax_val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model(str(out_dir / "model"))
    tok.save_pretrained(str(out_dir / "model"))

    print("SAVED:", out_dir / "model")
    print("AX val macro-F1:", metrics.get("eval_macro_f1"))

if __name__ == "__main__":
    main()
