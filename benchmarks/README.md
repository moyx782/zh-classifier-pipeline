# Clef Flash benchmark

This benchmark is intentionally separate from training. It compares the existing character TF-IDF + One-vs-Rest Logistic Regression baseline with a local Ollama `clef-flash` model on the same source-aware JSONL samples.

## What it measures

- **LR ↔ Clef agreement**: exact label-set agreement and mean/median Jaccard.
- **Coverage**: samples with at least one predicted label, plus how much of the 12-label taxonomy each model emits.
- **Per-class errors**: TP/FP/FN and precision/recall/F1 against the existing JSONL `cognitive_labels`.
- **Latency**: warm per-sample inference median / mean / p95. Baseline model load and Clef first warmup request are reported separately.

The JSONL labels are weak supervision, not human gold labels. Metrics against them are diagnostic only; model-vs-model agreement is reported separately for that reason.

## Dry run (no Ollama/model required)

```bash
python benchmarks/clef_flash_benchmark.py --dry-run
```

The fixture is synthetic and only verifies sampling/report plumbing. Its numbers must not be used to judge model quality.

## Live benchmark

Prerequisites:

```bash
pip install -r requirements.txt
python src/train_baseline.py  # only if the existing .joblib is missing
ollama --version              # must be >= 0.35.1
ollama pull clef-flash
```

Then run:

```bash
python benchmarks/clef_flash_benchmark.py --limit 24
```

By default the script:

1. reads the full source-aware JSONL if present, otherwise the 200-row preview;
2. selects a small deterministic sample, preferring coverage of the 12 cognitive labels;
3. loads the LR bundle once, so latency excludes model load;
4. sends all 12 labels to Clef Flash as independent `noul` questions in one `/v1/systemone` request per text;
5. applies the same `0.5` threshold to both models;
6. writes `outputs/benchmarks/clef_flash_samples.jsonl` and `outputs/benchmarks/clef_flash_benchmark.json`.

## Interpretation

Treat Clef Flash as a **secondary adjudicator candidate**, not a replacement baseline. Route only LR low-margin or ambiguous samples to Clef in a follow-up experiment, then check whether manual-review quality improves enough to justify the additional latency and memory cost.
