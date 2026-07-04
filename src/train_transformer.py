#!/usr/bin/env python3
"""train_transformer.py — Fine-tune a Chinese BERT for multi-label cognitive classification.

Reads:   outputs/universal_zh_classifier.source_aware.samples.jsonl
Writes:  outputs/models/source_aware_transformer/

Usage:
    pip install torch transformers scikit-learn
    python src/train_transformer.py --model bert-base-chinese --epochs 3 --batch-size 32

Optional flags:
    --min-confidence 0.72   只用 needs_review=false 的高置信样本训练（推荐）
    --max-samples 50000      限制样本数量（快速实验）
    --epochs 3
    --batch-size 32
    --lr 2e-5
    --output-dir outputs/models/source_aware_transformer
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_PATH = OUTPUTS_DIR / "universal_zh_classifier.source_aware.samples.jsonl"
DEFAULT_MODEL_DIR = OUTPUTS_DIR / "models" / "source_aware_transformer"

COGNITIVE_LABELS = [
    "ACTION_OPERATION", "TOOL_SYSTEM", "ERROR_PROBLEM", "STATE_STATUS",
    "DATA_INFO", "GOAL_INTENT", "METHOD_STRATEGY", "RELATION_STRUCTURE",
    "ENTITY_OBJECT", "ATTRIBUTE_PROPERTY", "CONCEPT_ABSTRACT", "LANGUAGE_EXPRESSION",
]


def load_samples(min_confidence=0.0, max_samples=None):
    samples = []
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("confidence", 0) < min_confidence:
                continue
            samples.append(rec)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chinese BERT for multi-label cognitive classification.")
    parser.add_argument("--model", type=str, default="bert-base-chinese",
                        help="HuggingFace model name or local path (default: bert-base-chinese)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64,
                        help="Max token length (词典词条短，64 够用)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="只用 confidence >= 此值的样本 (建议 0.72)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制样本数量（快速实验）")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    # ---- Check deps ----
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, hamming_loss
        import numpy as np
    except ImportError:
        print("ERROR: 缺少依赖。请先安装：")
        print("  pip install torch transformers scikit-learn numpy")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    samples = load_samples(
        min_confidence=args.min_confidence,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} samples (min_confidence={args.min_confidence})")
    if len(samples) < 100:
        print("ERROR: 样本太少")
        sys.exit(1)

    texts = [s["text"] for s in samples]
    label_matrix = []
    for s in samples:
        row = [1.0 if lb in s["cognitive_labels"] else 0.0 for lb in COGNITIVE_LABELS]
        label_matrix.append(row)
    label_matrix = np.array(label_matrix, dtype=np.float32)
    print(f"Label matrix shape: {label_matrix.shape}")

    # ---- Train/val split ----
    X_train, X_val, Y_train, Y_val = train_test_split(
        texts, label_matrix, test_size=args.val_split, random_state=42
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # ---- Tokenizer ----
    print(f"Loading tokenizer: {args.model}")
    tokenizer = BertTokenizer.from_pretrained(args.model)

    # ---- Dataset ----
    class JsonlDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            }

    train_ds = JsonlDataset(X_train, Y_train, tokenizer, args.max_length)
    val_ds   = JsonlDataset(X_val,   Y_val,   tokenizer, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ---- Model ----
    print(f"Loading model: {args.model}")
    bert = BertModel.from_pretrained(args.model)
    bert.to(device)

    class MultiLabelBert(nn.Module):
        def __init__(self, bert, num_labels):
            super().__init__()
            self.bert = bert
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = self.dropout(cls)
            logits = self.classifier(cls)
            return logits

    model = MultiLabelBert(bert, len(COGNITIVE_LABELS)).to(device)

    # ---- Loss & Optimizer ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # ---- Training loop ----
    best_val_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 200 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_dl)} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_dl)

        # ---- Validation ----
        model.eval()
        all_preds = []
        all_truths = []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()
                logits = model(input_ids, attention_mask).cpu().numpy()
                preds = (1 / (1 + np.exp(-logits))) > 0.5
                all_preds.append(preds)
                all_truths.append(labels)
        all_preds = np.vstack(all_preds)
        all_truths = np.vstack(all_truths)
        val_micro_f1 = f1_score(all_truths, all_preds, average="micro", zero_division=0)
        val_macro_f1 = f1_score(all_truths, all_preds, average="macro", zero_division=0)
        val_hamming = hamming_loss(all_truths, all_preds)

        print(f"\nEpoch {epoch}/{args.epochs}: train_loss={avg_loss:.4f}  "
              f"val_micro_f1={val_micro_f1:.4f}  val_macro_f1={val_macro_f1:.4f}  "
              f"hamming_loss={val_hamming:.4f}\n")

        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            save_dir = output_dir / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(save_dir)
            with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
                json.dump({"cognitive_labels": COGNITIVE_LABELS}, f, ensure_ascii=False, indent=2)
            with open(save_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "base_model": args.model,
                    "max_length": args.max_length,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "min_confidence": args.min_confidence,
                    "num_train": len(X_train),
                    "num_val": len(X_val),
                    "best_val_micro_f1": float(best_val_f1),
                    "best_val_macro_f1": float(val_macro_f1),
                    "best_val_hamming_loss": float(val_hamming),
                }, f, ensure_ascii=False, indent=2)
            print(f"  -> New best! Saved to {save_dir}")

    # ---- Save final ----
    save_dir = output_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"cognitive_labels": COGNITIVE_LABELS}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Best val micro-F1: {best_val_f1:.4f}")
    print(f"Best model:  {output_dir / 'best'}")
    print(f"Final model: {output_dir / 'final'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()