#!/usr/bin/env python3
"""train_baseline.py — Train a lightweight multi-label cognitive classifier baseline.

Model:
  TfidfVectorizer(analyzer="char", ngram_range=(1,4))
  + OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
  + MultiLabelBinarizer

Input:  outputs/universal_zh_classifier.source_aware.samples.jsonl
Output: outputs/models/source_aware_multilabel_classifier.joblib

No external LLM API calls; no Transformer fine-tuning.

Usage:
    python src/train_baseline.py
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_PATH = OUTPUTS_DIR / "universal_zh_classifier.source_aware.samples.jsonl"
MODEL_DIR = OUTPUTS_DIR / "models"
MODEL_PATH = MODEL_DIR / "source_aware_multilabel_classifier.joblib"


def load_samples():
    samples = []
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(rec)
    return samples


def main():
    # Check dependencies
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install scikit-learn joblib")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    if not samples:
        print("ERROR: No samples found at {}".format(SAMPLES_PATH))
        sys.exit(1)

    texts = [s["text"] for s in samples]
    label_sets = [s["cognitive_labels"] for s in samples]

    print(f"Loaded {len(samples)} samples")

    # Binarize labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_sets)
    label_names = mlb.classes_
    print(f"Cognitive labels: {len(label_names)}")
    for name in label_names:
        idx = list(mlb.classes_).index(name)
        count = Y[:, idx].sum()
        print(f"  {name}: {count}")

    # Vectorize
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    print(f"Feature matrix: {X.shape}")

    # Train classifier
    base_clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
    )
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X, Y)
    print("Training complete.")

    # Evaluate training accuracy (rough)
    from sklearn.metrics import classification_report, hamming_loss, f1_score
    Y_pred = clf.predict(X)
    train_hamming = hamming_loss(Y, Y_pred)
    train_micro_f1 = f1_score(Y, Y_pred, average="micro", zero_division=0)
    train_macro_f1 = f1_score(Y, Y_pred, average="macro", zero_division=0)
    print(f"Train hamming loss: {train_hamming:.4f}")
    print(f"Train micro-F1:     {train_micro_f1:.4f}")
    print(f"Train macro-F1:     {train_macro_f1:.4f}")

    # Save model bundle
    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "label_binarizer": mlb,
        "metadata": {
            "model_type": "TfidfVectorizer + OneVsRestClassifier(LogisticRegression)",
            "cognitive_labels": list(label_names),
            "num_samples": len(samples),
            "feature_dim": X.shape[1],
            "train_hamming_loss": float(train_hamming),
            "train_micro_f1": float(train_micro_f1),
            "train_macro_f1": float(train_macro_f1),
            "source_samples_path": str(SAMPLES_PATH),
        },
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Print classification report for top labels
    try:
        report = classification_report(Y, Y_pred, target_names=list(label_names), zero_division=0)
        print("\nClassification report (trainset):")
        print(report)
    except Exception:
        pass


if __name__ == "__main__":
    main()