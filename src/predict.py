#!/usr/bin/env python3
"""predict.py — Predict cognitive labels for input text using the trained baseline.

Usage:
    python src/predict.py --text "接口调用失败"
    python src/predict.py --text "Agent 调用工具失败"
    python src/predict.py --text "机器人路径规划"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUTS_DIR / "models" / "source_aware_multilabel_classifier.joblib"


def predict(text, model_path=MODEL_PATH):
    import joblib
    import numpy as np

    if not Path(model_path).exists():
        return {
            "error": f"Model not found at {model_path}. Run python src/train_baseline.py first.",
        }

    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    mlb = bundle["label_binarizer"]

    X = vectorizer.transform([text])
    # Get decision scores
    scores = classifier.decision_function(X)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    label_names = list(mlb.classes_)
    result = []
    for i, label in enumerate(label_names):
        score = float(scores[0][i])
        # Apply sigmoid to get probability-like score
        import math
        prob = 1.0 / (1.0 + math.exp(-max(min(score, 35), -35)))
        result.append({"label": label, "score": round(prob, 4)})

    # Sort by score descending
    result.sort(key=lambda x: -x["score"])

    # Predicted = score > 0.5 threshold
    predicted = [r for r in result if r["score"] > 0.5]

    return {
        "text": text,
        "predicted_cognitive_labels": predicted,
        "top_all": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict cognitive labels for input text.")
    parser.add_argument("--text", type=str, required=True, help="Input text to classify.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to model .joblib")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    args = parser.parse_args()

    result = predict(args.text, args.model)

    if args.json or "error" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Pretty print
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()