# Source-Aware Chinese Classifier Dataset & Baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moyx782/zh-classifier-pipeline/blob/master/train_colab.ipynb)

This project builds a **source-aware JSONL dataset** for a *general-purpose word/sentence
classifier* (通用词句分类器) based on GitHub open-source Chinese dictionary projects.

The project also trains a **lightweight multi-label cognitive-label classifier baseline**
and provides a prediction script.

---

## 1. Data Sources

### Primary source: `pwxcoo/chinese-xinhua`

- **GitHub**: <https://github.com/pwxcoo/chinese-xinhua>
- **Role**: primary bootstrap source for simplified Chinese vocabulary
- **Coverage**: words (ci), idioms (idiom), characters (word), xiehouyu (two-part allegorical sayings)
- **License hint**: Repo shows MIT license, but README says dictionary data was collected/scraped from the web.
  Suitable for research/prototyping; verify before commercial redistribution.
- **Why chosen as primary**:
  - Clean, structured JSON covering diverse Chinese word types.
  - Simplified Chinese, directly applicable for the target classifier.
  - MIT-licensed repository (though data provenance needs verification).

### Supplemental source: `g0v/moedict-data`

- **GitHub**: <https://github.com/g0v/moedict-data>
- **Role**: supplemental source to enrich definitions and multi-sense entries
- **Coverage**: Traditional / Taiwan Ministry of Education Mandarin dictionary
- **License hint**: Community-driven derivative of Taiwan MoE dictionary data; original dictionary
  text may have usage restrictions. Use as supplemental; do not redistribute raw text without verification.
- **Why chosen as supplemental**:
  - Rich multi-sense definitions from a high-quality academic dictionary.
  - Traditional Chinese; complements the simplified primary source without overwriting it.
  - Provides starting baseline now; labels are reinforced by keyword weak supervision.

### Candidate supplemental sources (catalog-only)

- `thunlp/THUOCL` — academic domain lexicons (IT, medical, financial, etc.) — for future domain enrichment.
- `liuhuanyong/DomainWordsDict` — domain-specific Chinese word lists — for future domain enrichment.

See `outputs/source_catalog.json` for full catalog; `outputs/source_inspection_report.md` for detailed inspection.

---

## 2. Project Structure

```
src/
  fetch_sources.py          Download raw data from GitHub
  inspect_sources.py       Inspect data sources, generate report
  build_dataset.py         Build source-aware JSONL dataset
  label_rules.py            Weak-supervision keyword rules
  train_baseline.py        Train baseline multi-label classifier
  predict.py                Predict cognitive labels for new text

outputs/
  source_catalog.json          Source catalog
  taxonomy.json                Cognitive + domain label taxonomy
  schema.sample.json           Sample JSONL record
  fetch_report.json            Fetch status report
  source_inspection_report.md  Source inspection report
  universal_zh_classifier.source_aware.samples.jsonl   Full dataset
  universal_zh_classifier.source_aware.preview.jsonl    Preview (first 200)
  build_report.md              Build report markdown
  build_report.json            Build report machine-readable
  models/
    source_aware_multilabel_classifier.joblib    Trained baseline model

raw/
  pwxcoo__chinese-xinhua/    Downloaded raw JSON files
  g0v__moedict-data/         Supplemental raw files (if fetched)
```

---

## 3. JSONL Schema

Each line in `universal_zh_classifier.source_aware.samples.jsonl` has:

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | The primary text being labeled |
| `normalized_text` | str | Whitespace-normalised text |
| `text_type` | str | e.g. `word_or_phrase`, `idiom`, `character`, `xiehouyu_riddle`, `xiehouyu_answer`, `xiehouyu_pair`, `example_sentence`, `technical_phrase` |
| `cognitive_labels` | list[str] | Cognitive-level labels (see `taxonomy.json`) |
| `domain_labels` | list[str] | Domain-level labels (see `taxonomy.json`) |
| `confidence` | float | 0.0–0.95; derived from source base + keyword bonus |
| `needs_review` | bool | `True` if `confidence < 0.72` |
| `source_repo` | str | GitHub repo URL |
| `source_repo_key` | str | e.g. `pwxcoo/chinese-xinhua` |
| `source_dataset` | str | e.g. `ci`, `idiom`, `word`, `xiehouyu`, `dict-revised` |
| `source_record_id` | str | Stable traceable ID, e.g. `ci:123`, `idiom:55:word`, `xiehouyu:9:pair` |
| `license_hint` | str | Redistributability / license guidance — every sample carries this |
| `raw_fields` | dict | Original raw fields from the source record (truncated to 500 chars max) |
| `label_evidence` | dict | How each label was matched (keywords hit) |
| `labeling_method` | str | e.g. `source_default_labels + keyword_weak_supervision` |
| `labels_note` | str | Human-readable note on labeling approach |
| `generator_version` | str | Generator tag (`source-aware-jsonl-v2`) |

### `raw_fields` significance

`raw_fields` is **not just the text** — it preserves the original fields from the source dictionary
(definition, pinyin, strokes, radicals, derivation, example, abbreviation, more, etc.).
This allows for:
- later re-labeling with stronger rules;
- traceability of how labels were derived;
- filtering by source-specific fields;
- bias detection across sources.

### `license_hint` significance

Every sample carries a `license_hint` string tracing back to the source's redistribution guidance.
This is a non-legal-nudge field for users to understand what they can do with the data; always
verify against the upstream repository for commercial use.

### `needs_review` significance

Set to `True` when `confidence < 0.72`. This flags samples where the weak-supervision labels
are probably noisy and should be manually reviewed before being treated as gold labels.

### `taxonomy` — Cognitive Labels

- `ACTION_OPERATION`, `TOOL_SYSTEM`, `ERROR_PROBLEM`, `STATE_STATUS`, `DATA_INFO`,
  `GOAL_INTENT`, `METHOD_STRATEGY`, `RELATION_STRUCTURE`, `ENTITY_OBJECT`,
  `ATTRIBUTE_PROPERTY`, `CONCEPT_ABSTRACT`, `LANGUAGE_EXPRESSION`

### `taxonomy` — Domain Labels

- `GENERAL`, `CHINESE_DICTIONARY`, `SOFTWARE`, `ML`, `ROBOTICS`, `NETWORK`, `AGENT_TOOLING`

---

## 4. How to Download Data

```bash
python src/fetch_sources.py
```

This pulls:
- `pwxcoo/chinese-xinhua` → `raw/pwxcoo__chinese-xinhua/`
- `g0v/moedict-data` → `raw/g0v__moedict-data/` (optional)

A `fetch_report.json` is written to `outputs/`.

### Inspect

```bash
python src/inspect_sources.py
```

Generates `outputs/source_inspection_report.md` with license, scope, fields, quality
and commercial-risk assessments for each cataloged source.

---

## 5. How to Generate JSONL

```bash
python src/build_dataset.py
```

This reads the raw files and:
- Applies per-source default labels (see spec rules in `src/build_dataset.py`).
- Applies weak-supervision keyword rules from `src/label_rules.py`.
- Truncates long fields to 500 characters.
- Deduplicates by `(text, cognitive_labels, domain_labels, source_repo_key, source_dataset)` keeping
  the higher-confidence sample on conflict.
- Writes `outputs/universal_zh_classifier.source_aware.samples.jsonl` and a 200-record preview.
- Writes `outputs/build_report.md` and `outputs/build_report.json`.

To inspect labeling rules:

```bash
python src/label_rules.py
```

---

## 6. How to Train

```bash
pip install -r requirements.txt
python src/train_baseline.py
```

Trains:
- `TfidfVectorizer(analyzer="char", ngram_range=(1,4))`
- `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))`
- Uses `MultiLabelBinarizer` to binarize `cognitive_labels`

Model is saved to `outputs/models/source_aware_multilabel_classifier.joblib`.

No external LLM API calls; no Transformer fine-tuning. This is a lightweight baseline.

---

## 7. How to Predict

```bash
python src/predict.py --text "接口调用失败"
python src/predict.py --text "Agent 调用工具失败"
python src/predict.py --text "机器人路径规划"
```

Returns JSON with `predicted_cognitive_labels` (score > 0.5) and `top_all` (all labels with sigmoid probabilities).

---

## 8. One-click Run

### Bash

```bash
bash run_all.sh
```

### Windows PowerShell

```powershell
.\run_all.ps1
```

These will:
1. Inspect sources.
2. Fetch raw data.
3. Build the JSONL dataset.
4. Train the baseline model.
5. Run three prediction examples.

---

## 9. Disclaimer

**This dataset is generated with weak supervision** (source-default labels + keyword rules).

It is **not human gold-standard** data. Labels are noisy and should be reviewed via the
`needs_review` flag before production use.

Check `license_hint` on every sample and the upstream repo before redistributing any raw fields.

`generator_version`: `source-aware-jsonl-v2`