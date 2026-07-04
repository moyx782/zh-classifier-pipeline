#!/usr/bin/env python3
"""predict_transformer.py — Predict cognitive labels using the fine-tuned Transformer.

Usage:
    python src/predict_transformer.py --text "接口调用失败"
    python src/predict_transformer.py --text "机器人路径规划" --model-dir outputs/models/source_aware_transformer/best
"""

import argparse
import json
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = OUTPUTS_DIR = (
    PROJECT_ROOT / "outputs" / "models" / "source_aware_transformer" / "best"
)

COGNITIVE_LABELS = [
    "ACTION_OPERATION", "TOOL_SYSTEM", "ERROR_PROBLEM", "STATE_STATUS",
    "DATA_INFO", "GOAL_INTENT", "METHOD_STRATEGY", "RELATION_STRUCTURE",
    "ENTITY_OBJECT", "ATTRIBUTE_PROPERTY", "CONCEPT_ABSTRACT", "LANGUAGE_EXPRESSION",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: 模型目录不存在: {model_dir}")
        print("请先运行: python src/train_transformer.py")
        sys.exit(1)

    try:
        import torch
        import numpy as np
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("ERROR: pip install torch transformers numpy")
        sys.exit(1)

    import torch.nn as nn

    class MultiLabelBert(nn.Module):
        def __init__(self, bert, num_labels):
            super().__init__()
            self.bert = bert
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(bert.config.hidden_size, num_labels)
        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(cls))

    labels_file = model_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            label_cfg = json.load(f)
        labels = label_cfg.get("cognitive_labels", COGNITIVE_LABELS)
    else:
        labels = COGNITIVE_LABELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    bert = BertModel.from_pretrained(
        json.load(open(model_dir / "config.json", encoding="utf-8"))["base_model"]
    )
    model = MultiLabelBert(bert, len(labels))
    model.load_state_dict(torch.load(model_dir / "pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()

    enc = tokenizer(args.text, truncation=True, padding="max_length",
                    max_length=args.max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask).cpu().numpy()[0]

    result = []
    for i, label in enumerate(labels):
        prob = 1.0 / (1.0 + math.exp(-max(min(logits[i], 35), -35)))
        result.append({"label": label, "score": round(float(prob), 4)})

    result.sort(key=lambda x: -x["score"])
    predicted = [r for r in result if r["score"] > 0.5]

    output = {"text": args.text, "predicted_cognitive_labels": predicted, "top_all": result}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()