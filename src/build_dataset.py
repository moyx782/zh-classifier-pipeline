#!/usr/bin/env python3
"""build_dataset.py — Build source-aware JSONL dataset from raw dictionary data.

Reads raw/*.json files downloaded by fetch_sources.py and the source_catalog.json.
For each record, generates one or more JSONL samples per the rules in the
specification, applying source-default labels + keyword weak supervision.
Saves to:
  outputs/universal_zh_classifier.source_aware.samples.jsonl
  outputs/universal_zh_classifier.source_aware.preview.jsonl      (first 200)
  outputs/build_report.md
  outputs/build_report.json

Usage:
    python src/build_dataset.py
"""

import json
import os
import re
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_rules import apply_weak_supervision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = PROJECT_ROOT / "raw"
CATALOG_PATH = OUTPUTS_DIR / "source_catalog.json"
SAMPLES_PATH = OUTPUTS_DIR / "universal_zh_classifier.source_aware.samples.jsonl"
PREVIEW_PATH = OUTPUTS_DIR / "universal_zh_classifier.source_aware.preview.jsonl"
BUILD_REPORT_MD = OUTPUTS_DIR / "build_report.md"
BUILD_REPORT_JSON = OUTPUTS_DIR / "build_report.json"

GENERATOR_VERSION = "source-aware-jsonl-v2"
TRUNCATE_LEN = 500
REVIEW_THRESHOLD = 0.72


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text, maxlen=TRUNCATE_LEN):
    if not text:
        return ""
    if len(text) <= maxlen:
        return text
    return text[:maxlen] + "..."


def normalize_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_record_id(*parts):
    """Build a stable source_record_id from parts, e.g. ci:123, idiom:55:word."""
    return ":".join(str(p) for p in parts)


def truncate_raw_fields(raw_fields):
    """Truncate long string fields in raw_fields dict to TRUNCATE_LEN."""
    out = {}
    for k, v in raw_fields.items():
        if isinstance(v, str):
            out[k] = truncate(v)
        elif isinstance(v, list):
            out[k] = [truncate(str(item)) for item in v]
        else:
            out[k] = v
    return out


def dedup_key(sample):
    return (
        sample["text"],
        tuple(sorted(sample["cognitive_labels"])),
        tuple(sorted(sample["domain_labels"])),
        sample["source_repo_key"],
        sample["source_dataset"],
    )


# ---- Per-source builders ---------------------------------------------------

def build_ci_samples(records, catalog_info, repo_key):
    """Build samples from ci.json (word dictionary entries)."""
    samples = []
    license_hint = catalog_info.get("license_hint", "")
    source_repo = catalog_info.get("source_repo", "")
    source_dataset = "ci"

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        ci = rec.get("ci") or rec.get("word") or ""
        explanation = rec.get("explanation") or ""
        text = ci.strip()
        if not text:
            continue

        raw_fields = {
            "ci": truncate(ci),
            "explanation": truncate(explanation),
        }
        raw_fields = truncate_raw_fields(raw_fields)

        ws = apply_weak_supervision(
            text=text,
            raw_fields=raw_fields,
            base_cognitive=["LANGUAGE_EXPRESSION", "DATA_INFO"],
            base_domain=["CHINESE_DICTIONARY"],
            base_confidence=0.5,
        )

        sample = {
            "text": text,
            "normalized_text": normalize_text(text),
            "text_type": "word_or_phrase",
            "cognitive_labels": ws["cognitive_labels"],
            "domain_labels": ws["domain_labels"],
            "confidence": ws["confidence"],
            "needs_review": ws["confidence"] < REVIEW_THRESHOLD,
            "source_repo": source_repo,
            "source_repo_key": repo_key,
            "source_dataset": source_dataset,
            "source_record_id": make_record_id("ci", idx),
            "license_hint": license_hint,
            "raw_fields": raw_fields,
            "label_evidence": ws["label_evidence"],
            "labeling_method": ws["labeling_method"],
            "labels_note": "Base labels: LANGUAGE_EXPRESSION + DATA_INFO / CHINESE_DICTIONARY; augmented by keyword weak supervision.",
            "generator_version": GENERATOR_VERSION,
        }
        samples.append(sample)
    return samples


def build_idiom_samples(records, catalog_info, repo_key):
    """Build samples from idiom.json."""
    samples = []
    license_hint = catalog_info.get("license_hint", "")
    source_repo = catalog_info.get("source_repo", "")
    source_dataset = "idiom"

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        word = rec.get("word") or ""
        pinyin = rec.get("pinyin") or ""
        abbreviation = rec.get("abbreviation") or ""
        explanation = rec.get("explanation") or ""
        derivation = rec.get("derivation") or ""
        example = rec.get("example") or ""

        text = word.strip()
        if not text:
            continue

        raw_fields = {
            "word": truncate(word),
            "pinyin": truncate(pinyin),
            "abbreviation": truncate(abbreviation),
            "explanation": truncate(explanation),
            "derivation": truncate(derivation),
            "example": truncate(example),
        }
        raw_fields = truncate_raw_fields(raw_fields)

        ws = apply_weak_supervision(
            text=text,
            raw_fields=raw_fields,
            base_cognitive=["LANGUAGE_EXPRESSION", "CONCEPT_ABSTRACT"],
            base_domain=["CHINESE_DICTIONARY"],
            base_confidence=0.5,
        )

        sample = {
            "text": text,
            "normalized_text": normalize_text(text),
            "text_type": "idiom",
            "cognitive_labels": ws["cognitive_labels"],
            "domain_labels": ws["domain_labels"],
            "confidence": ws["confidence"],
            "needs_review": ws["confidence"] < REVIEW_THRESHOLD,
            "source_repo": source_repo,
            "source_repo_key": repo_key,
            "source_dataset": source_dataset,
            "source_record_id": make_record_id("idiom", idx, "word"),
            "license_hint": license_hint,
            "raw_fields": raw_fields,
            "label_evidence": ws["label_evidence"],
            "labeling_method": ws["labeling_method"],
            "labels_note": "Base labels: LANGUAGE_EXPRESSION + CONCEPT_ABSTRACT / CHINESE_DICTIONARY; augmented by keyword weak supervision.",
            "generator_version": GENERATOR_VERSION,
        }
        samples.append(sample)

        # Extra example_sentence sample
        if example and example.strip():
            ex_text = example.strip()
            ex_raw = {"example": truncate(ex_text), "word": truncate(word)}
            ex_raw = truncate_raw_fields(ex_raw)

            ws_ex = apply_weak_supervision(
                text=ex_text,
                raw_fields=ex_raw,
                base_cognitive=["LANGUAGE_EXPRESSION"],
                base_domain=["CHINESE_DICTIONARY"],
                base_confidence=0.45,
            )

            ex_sample = {
                "text": ex_text,
                "normalized_text": normalize_text(ex_text),
                "text_type": "example_sentence",
                "cognitive_labels": ws_ex["cognitive_labels"],
                "domain_labels": ws_ex["domain_labels"],
                "confidence": ws_ex["confidence"],
                "needs_review": ws_ex["confidence"] < REVIEW_THRESHOLD,
                "source_repo": source_repo,
                "source_repo_key": repo_key,
                "source_dataset": source_dataset,
                "source_record_id": make_record_id("idiom", idx, "example"),
                "license_hint": license_hint,
                "raw_fields": ex_raw,
                "label_evidence": ws_ex["label_evidence"],
                "labeling_method": ws_ex["labeling_method"],
                "labels_note": "Example sentence extracted from idiom record; base label LANGUAGE_EXPRESSION; weak supervision applied.",
                "generator_version": GENERATOR_VERSION,
            }
            samples.append(ex_sample)
    return samples


def build_word_samples(records, catalog_info, repo_key):
    """Build samples from word.json (individual characters)."""
    samples = []
    license_hint = catalog_info.get("license_hint", "")
    source_repo = catalog_info.get("source_repo", "")
    source_dataset = "word"

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        word = rec.get("word") or ""
        oldword = rec.get("oldword") or ""
        strokes = rec.get("strokes") or ""
        pinyin = rec.get("pinyin") or ""
        radicals = rec.get("radicals") or ""
        explanation = rec.get("explanation") or ""
        more = rec.get("more") or ""

        text = word.strip()
        if not text:
            continue

        raw_fields = {
            "word": truncate(word),
            "oldword": truncate(oldword),
            "strokes": truncate(str(strokes)) if strokes is not None else "",
            "pinyin": truncate(pinyin),
            "radicals": truncate(radicals),
            "explanation": truncate(explanation),
            "more": truncate(more),
        }
        raw_fields = truncate_raw_fields(raw_fields)

        base_cog = ["LANGUAGE_EXPRESSION", "ENTITY_OBJECT"]
        # Check for strokes/radicals/pinyin => add ATTRIBUTE_PROPERTY
        has_attr = any([
            strokes not in (None, "", 0, "0"),
            radicals not in (None, ""),
            pinyin not in (None, ""),
        ])
        # apply doesn't automatically add ATTRIBUTE_PROPERTY unless matched by keywords,
        # but strokes/radicals/pinyin are keyword triggers so weak supervision should catch them.
        ws = apply_weak_supervision(
            text=text,
            raw_fields=raw_fields,
            base_cognitive=base_cog,
            base_domain=["CHINESE_DICTIONARY"],
            base_confidence=0.5,
        )
        # Ensure ATTRIBUTE_PROPERTY if has_attr and not already present
        if has_attr and "ATTRIBUTE_PROPERTY" not in ws["cognitive_labels"]:
            ws["cognitive_labels"] = sorted(ws["cognitive_labels"] + ["ATTRIBUTE_PROPERTY"])
            ws["label_evidence"].setdefault("ATTRIBUTE_PROPERTY", []).append("strokes/radicals/pinyin present")
        # Re-calc confidence
        total_hits = sum(len(v) for v in ws["label_evidence"].values())
        ws["confidence"] = round(min(0.5 + total_hits * 0.04, 0.95), 4)
        ws["needs_review"] = ws["confidence"] < REVIEW_THRESHOLD

        sample = {
            "text": text,
            "normalized_text": normalize_text(text),
            "text_type": "character",
            "cognitive_labels": ws["cognitive_labels"],
            "domain_labels": ws["domain_labels"],
            "confidence": ws["confidence"],
            "needs_review": ws["needs_review"],
            "source_repo": source_repo,
            "source_repo_key": repo_key,
            "source_dataset": source_dataset,
            "source_record_id": make_record_id("word", idx, "word"),
            "license_hint": license_hint,
            "raw_fields": raw_fields,
            "label_evidence": ws["label_evidence"],
            "labeling_method": ws["labeling_method"],
            "labels_note": "Base labels: LANGUAGE_EXPRESSION + ENTITY_OBJECT / CHINESE_DICTIONARY; ATTRIBUTE_PROPERTY added if strokes/radicals/pinyin present; weak supervision applied.",
            "generator_version": GENERATOR_VERSION,
        }
        samples.append(sample)
    return samples


def build_xiehouyu_samples(records, catalog_info, repo_key):
    """Build samples from xiehouyu.json — generates riddle, answer, and pair samples."""
    samples = []
    license_hint = catalog_info.get("license_hint", "")
    source_repo = catalog_info.get("source_repo", "")
    source_dataset = "xiehouyu"

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        riddle = rec.get("riddle") or ""
        answer = rec.get("answer") or ""

        # Riddle sample
        if riddle.strip():
            raw_fields = {"riddle": truncate(riddle), "answer": truncate(answer)}
            raw_fields = truncate_raw_fields(raw_fields)
            ws_r = apply_weak_supervision(
                text=riddle.strip(),
                raw_fields=raw_fields,
                base_cognitive=["LANGUAGE_EXPRESSION", "RELATION_STRUCTURE"],
                base_domain=["CHINESE_DICTIONARY"],
                base_confidence=0.5,
            )
            r_sample = {
                "text": riddle.strip(),
                "normalized_text": normalize_text(riddle.strip()),
                "text_type": "xiehouyu_riddle",
                "cognitive_labels": ws_r["cognitive_labels"],
                "domain_labels": ws_r["domain_labels"],
                "confidence": ws_r["confidence"],
                "needs_review": ws_r["confidence"] < REVIEW_THRESHOLD,
                "source_repo": source_repo,
                "source_repo_key": repo_key,
                "source_dataset": source_dataset,
                "source_record_id": make_record_id("xiehouyu", idx, "riddle"),
                "license_hint": license_hint,
                "raw_fields": raw_fields,
                "label_evidence": ws_r["label_evidence"],
                "labeling_method": ws_r["labeling_method"],
                "labels_note": "Base labels: LANGUAGE_EXPRESSION + RELATION_STRUCTURE / CHINESE_DICTIONARY; weak supervision applied.",
                "generator_version": GENERATOR_VERSION,
            }
            samples.append(r_sample)

        # Answer sample
        if answer.strip():
            raw_fields = {"riddle": truncate(riddle), "answer": truncate(answer)}
            raw_fields = truncate_raw_fields(raw_fields)
            ws_a = apply_weak_supervision(
                text=answer.strip(),
                raw_fields=raw_fields,
                base_cognitive=["LANGUAGE_EXPRESSION", "RELATION_STRUCTURE"],
                base_domain=["CHINESE_DICTIONARY"],
                base_confidence=0.5,
            )
            a_sample = {
                "text": answer.strip(),
                "normalized_text": normalize_text(answer.strip()),
                "text_type": "xiehouyu_answer",
                "cognitive_labels": ws_a["cognitive_labels"],
                "domain_labels": ws_a["domain_labels"],
                "confidence": ws_a["confidence"],
                "needs_review": ws_a["confidence"] < REVIEW_THRESHOLD,
                "source_repo": source_repo,
                "source_repo_key": repo_key,
                "source_dataset": source_dataset,
                "source_record_id": make_record_id("xiehouyu", idx, "answer"),
                "license_hint": license_hint,
                "raw_fields": raw_fields,
                "label_evidence": ws_a["label_evidence"],
                "labeling_method": ws_a["labeling_method"],
                "labels_note": "Base labels: LANGUAGE_EXPRESSION + RELATION_STRUCTURE / CHINESE_DICTIONARY; weak supervision applied.",
                "generator_version": GENERATOR_VERSION,
            }
            samples.append(a_sample)

        # Pair sample: riddle + answer
        if riddle.strip() and answer.strip():
            pair_text = f"{riddle.strip()} —— {answer.strip()}"
            raw_fields = {"riddle": truncate(riddle), "answer": truncate(answer)}
            raw_fields = truncate_raw_fields(raw_fields)
            ws_p = apply_weak_supervision(
                text=pair_text,
                raw_fields=raw_fields,
                base_cognitive=["LANGUAGE_EXPRESSION", "RELATION_STRUCTURE", "DATA_INFO"],
                base_domain=["CHINESE_DICTIONARY"],
                base_confidence=0.52,
            )
            p_sample = {
                "text": pair_text,
                "normalized_text": normalize_text(pair_text),
                "text_type": "xiehouyu_pair",
                "cognitive_labels": ws_p["cognitive_labels"],
                "domain_labels": ws_p["domain_labels"],
                "confidence": ws_p["confidence"],
                "needs_review": ws_p["confidence"] < REVIEW_THRESHOLD,
                "source_repo": source_repo,
                "source_repo_key": repo_key,
                "source_dataset": source_dataset,
                "source_record_id": make_record_id("xiehouyu", idx, "pair"),
                "license_hint": license_hint,
                "raw_fields": raw_fields,
                "label_evidence": ws_p["label_evidence"],
                "labeling_method": ws_p["labeling_method"],
                "labels_note": "Base labels: LANGUAGE_EXPRESSION + RELATION_STRUCTURE + DATA_INFO / CHINESE_DICTIONARY; pair sample; weak supervision applied.",
                "generator_version": GENERATOR_VERSION,
            }
            samples.append(p_sample)
    return samples


# ---- Main ------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()

    all_samples = []
    build_stats = {
        "sources_processed": [],
        "total_samples": 0,
        "source_repo_dataset_counts": defaultdict(int),
        "text_type_counts": defaultdict(int),
        "cognitive_label_counts": defaultdict(int),
        "domain_label_counts": defaultdict(int),
        "needs_review_count": 0,
        "confidence_stats": {"min": 1.0, "max": 0.0, "avg": 0.0, "sum": 0.0},
        "raw_files_processed": [],
    }

    # Map repo to raw dir
    raw_dirs = {
        "pwxcoo/chinese-xinhua": RAW_DIR / "pwxcoo__chinese-xinhua",
        "g0v/moedict-data": RAW_DIR / "g0v__moedict-data",
    }

    # Process primary source (pwxcoo/chinese-xinhua)
    primary_repo = "pwxcoo/chinese-xinhua"
    if primary_repo in catalog:
        info = catalog[primary_repo]
        raw_dir = raw_dirs.get(primary_repo, RAW_DIR / primary_repo.replace("/", "__"))

        # ci.json
        ci_path = raw_dir / "ci.json"
        if ci_path.exists():
            records = load_json(ci_path)
            samples = build_ci_samples(records, info, primary_repo)
            all_samples.extend(samples)
            build_stats["raw_files_processed"].append({"file": "ci.json", "records": len(records), "samples": len(samples)})
            print(f"ci.json: {len(records)} records -> {len(samples)} samples")
        else:
            print(f"WARN: {ci_path} not found")

        # idiom.json
        idiom_path = raw_dir / "idiom.json"
        if idiom_path.exists():
            records = load_json(idiom_path)
            samples = build_idiom_samples(records, info, primary_repo)
            all_samples.extend(samples)
            build_stats["raw_files_processed"].append({"file": "idiom.json", "records": len(records), "samples": len(samples)})
            print(f"idiom.json: {len(records)} records -> {len(samples)} samples")
        else:
            print(f"WARN: {idiom_path} not found")

        # word.json
        word_path = raw_dir / "word.json"
        if word_path.exists():
            records = load_json(word_path)
            samples = build_word_samples(records, info, primary_repo)
            all_samples.extend(samples)
            build_stats["raw_files_processed"].append({"file": "word.json", "records": len(records), "samples": len(samples)})
            print(f"word.json: {len(records)} records -> {len(samples)} samples")
        else:
            print(f"WARN: {word_path} not found")

        # xiehouyu.json
        xiehouyu_path = raw_dir / "xiehouyu.json"
        if xiehouyu_path.exists():
            records = load_json(xiehouyu_path)
            samples = build_xiehouyu_samples(records, info, primary_repo)
            all_samples.extend(samples)
            build_stats["raw_files_processed"].append({"file": "xiehouyu.json", "records": len(records), "samples": len(samples)})
            print(f"xiehouyu.json: {len(records)} records -> {len(samples)} samples")
        else:
            print(f"WARN: {xiehouyu_path} not found")

        build_stats["sources_processed"].append(primary_repo)

    # Process supplemental source (g0v/moedict-data) if available
    supplemental_repo = "g0v/moedict-data"
    if supplemental_repo in catalog:
        info = catalog[supplemental_repo]
        raw_dir = raw_dirs.get(supplemental_repo, RAW_DIR / supplemental_repo.replace("/", "__"))
        moedict_path = raw_dir / "dict-revised.json"
        if moedict_path.exists():
            try:
                records = load_json(moedict_path)
                # For moedict, generate simplified supplemental samples (title-based)
                samples = []
                if isinstance(records, dict):
                    # dict-revised.json may be { title: {...} }
                    for idx, (title, body) in enumerate(records.items()):
                        if not title or not title.strip():
                            continue
                        text = title.strip()
                        raw_fields = {"title": truncate(text)}
                        heteronyms = body.get("heteronyms", []) if isinstance(body, dict) else []
                        if isinstance(heteronyms, list) and heteronyms:
                            for h in heteronyms[:1]:
                                if isinstance(h, dict):
                                    raw_fields["pinyin"] = truncate(str(h.get("pinyin", "")))
                                    defs = h.get("definitions", [])
                                    def_texts = []
                                    if isinstance(defs, list):
                                        for d in defs[:3]:
                                            if isinstance(d, dict):
                                                def_texts.append(str(d.get("def", "")))
                                    raw_fields["definition"] = truncate(" ".join(def_texts))
                        raw_fields = truncate_raw_fields(raw_fields)
                        ws = apply_weak_supervision(
                            text=text,
                            raw_fields=raw_fields,
                            base_cognitive=["LANGUAGE_EXPRESSION", "DATA_INFO"],
                            base_domain=["CHINESE_DICTIONARY"],
                            base_confidence=0.4,
                        )
                        sample = {
                            "text": text,
                            "normalized_text": normalize_text(text),
                            "text_type": "word_or_phrase",
                            "cognitive_labels": ws["cognitive_labels"],
                            "domain_labels": ws["domain_labels"],
                            "confidence": ws["confidence"],
                            "needs_review": ws["confidence"] < REVIEW_THRESHOLD,
                            "source_repo": info.get("source_repo", ""),
                            "source_repo_key": supplemental_repo,
                            "source_dataset": "dict-revised",
                            "source_record_id": make_record_id("dict-revised", idx, "title"),
                            "license_hint": info.get("license_hint", ""),
                            "raw_fields": raw_fields,
                            "label_evidence": ws["label_evidence"],
                            "labeling_method": ws["labeling_method"],
                            "labels_note": "Supplemental source; base labels: LANGUAGE_EXPRESSION + DATA_INFO / CHINESE_DICTIONARY; weak supervision applied.",
                            "generator_version": GENERATOR_VERSION,
                        }
                        samples.append(sample)
                        if idx >= 5000:
                            break
                all_samples.extend(samples)
                build_stats["raw_files_processed"].append({"file": "dict-revised.json", "records": len(records) if isinstance(records, (list, dict)) else 0, "samples": len(samples)})
                print(f"dict-revised.json: {len(samples)} supplemental samples")
            except Exception as e:
                print(f"WARN: Error processing moedict data: {e}")
        else:
            print(f"INFO: {moedict_path} not found (optional)")
        build_stats["sources_processed"].append(supplemental_repo)

    # Dedup: keep higher confidence version for duplicate keys
    dedup_map = {}
    for s in all_samples:
        key = dedup_key(s)
        if key in dedup_map:
            if s["confidence"] > dedup_map[key]["confidence"]:
                dedup_map[key] = s
        else:
            dedup_map[key] = s
    all_samples = list(dedup_map.values())

    # Stats
    conf_vals = []
    for s in all_samples:
        key = f"{s['source_repo_key']}/{s['source_dataset']}"
        build_stats["source_repo_dataset_counts"][key] += 1
        build_stats["text_type_counts"][s["text_type"]] += 1
        for cl in s["cognitive_labels"]:
            build_stats["cognitive_label_counts"][cl] += 1
        for dl in s["domain_labels"]:
            build_stats["domain_label_counts"][dl] += 1
        if s["needs_review"]:
            build_stats["needs_review_count"] += 1
        conf_vals.append(s["confidence"])

    if conf_vals:
        build_stats["confidence_stats"]["min"] = round(min(conf_vals), 4)
        build_stats["confidence_stats"]["max"] = round(max(conf_vals), 4)
        build_stats["confidence_stats"]["avg"] = round(sum(conf_vals) / len(conf_vals), 4)

    build_stats["total_samples"] = len(all_samples)
    build_stats["after_dedup"] = len(all_samples)

    # Write samples JSONL
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write preview (first 200)
    with open(PREVIEW_PATH, "w", encoding="utf-8") as f:
        for s in all_samples[:200]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Build report (JSON)
    json_report = {
        "total_samples": build_stats["total_samples"],
        "after_dedup": build_stats["after_dedup"],
        "sources_processed": build_stats["sources_processed"],
        "needs_review_count": build_stats["needs_review_count"],
        "confidence_stats": build_stats["confidence_stats"],
        "source_repo_dataset_counts": dict(build_stats["source_repo_dataset_counts"]),
        "text_type_counts": dict(build_stats["text_type_counts"]),
        "cognitive_label_counts": dict(build_stats["cognitive_label_counts"]),
        "domain_label_counts": dict(build_stats["domain_label_counts"]),
        "raw_files_processed": build_stats["raw_files_processed"],
        "review_threshold": REVIEW_THRESHOLD,
        "generator_version": GENERATOR_VERSION,
        "truncate_len": TRUNCATE_LEN,
    }
    with open(BUILD_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    # Build report (MD)
    md_lines = []
    md_lines.append("# Build Report — Source-Aware JSONL Dataset")
    md_lines.append("")
    md_lines.append(f"*Generated by build_dataset.py v{GENERATOR_VERSION}*")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- **Total samples**: {build_stats['total_samples']}")
    md_lines.append(f"- **Needs review** (confidence < {REVIEW_THRESHOLD}): {build_stats['needs_review_count']}")
    md_lines.append(f"- **Confidence**: min={build_stats['confidence_stats']['min']}, max={build_stats['confidence_stats']['max']}, avg={build_stats['confidence_stats']['avg']}")
    md_lines.append("")
    md_lines.append("## Source Distribution")
    md_lines.append("")
    md_lines.append("| Source / Dataset | Count |")
    md_lines.append("|------------------|-------|")
    for key, count in sorted(build_stats["source_repo_dataset_counts"].items()):
        md_lines.append(f"| {key} | {count} |")
    md_lines.append("")
    md_lines.append("## Text Type Distribution")
    md_lines.append("")
    md_lines.append("| text_type | Count |")
    md_lines.append("|-----------|-------|")
    for key, count in sorted(build_stats["text_type_counts"].items()):
        md_lines.append(f"| {key} | {count} |")
    md_lines.append("")
    md_lines.append("## Cognitive Label Distribution")
    md_lines.append("")
    md_lines.append("| Label | Count |")
    md_lines.append("|-------|-------|")
    for key, count in sorted(build_stats["cognitive_label_counts"].items(), key=lambda x: -x[1]):
        md_lines.append(f"| {key} | {count} |")
    md_lines.append("")
    md_lines.append("## Domain Label Distribution")
    md_lines.append("")
    md_lines.append("| Label | Count |")
    md_lines.append("|-------|-------|")
    for key, count in sorted(build_stats["domain_label_counts"].items(), key=lambda x: -x[1]):
        md_lines.append(f"| {key} | {count} |")
    md_lines.append("")
    md_lines.append("## Raw Files Processed")
    md_lines.append("")
    for ri in build_stats["raw_files_processed"]:
        md_lines.append(f"- `{ri['file']}`: {ri['records']} records -> {ri['samples']} samples")
    md_lines.append("")
    md_lines.append("## Deduplication")
    md_lines.append("")
    md_lines.append("Dedup key = `text + cognitive_labels + domain_labels + source_repo_key + source_dataset`.")
    md_lines.append("On duplicate keys, the sample with the higher `confidence` is retained.")
    md_lines.append("")
    md_lines.append("## Disclaimer")
    md_lines.append("")
    md_lines.append("This dataset is generated with **weak supervision** (source-default labels + keyword rules).")
    md_lines.append("It is **not** human gold-standard data. All samples should be reviewed before use in")
    md_lines.append("production. Refer to `license_hint` on each sample for redistribution guidance.")
    md_lines.append("")

    with open(BUILD_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Print summary
    print("=" * 60)
    print("build_dataset.py — Dataset build complete")
    print("=" * 60)
    print(f"Total samples: {build_stats['total_samples']}")
    print(f"Needs review: {build_stats['needs_review_count']}")
    print(f"Confidence: min={build_stats['confidence_stats']['min']}, max={build_stats['confidence_stats']['max']}, avg={build_stats['confidence_stats']['avg']}")
    print(f"\nSource / Dataset distribution:")
    for key, count in sorted(build_stats["source_repo_dataset_counts"].items()):
        print(f"  {key}: {count}")
    print(f"\nCognitive label distribution:")
    for key, count in sorted(build_stats["cognitive_label_counts"].items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")
    print(f"\nDomain label distribution:")
    for key, count in sorted(build_stats["domain_label_counts"].items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")
    print(f"\nSamples JSONL:     {SAMPLES_PATH}")
    print(f"Preview JSONL:      {PREVIEW_PATH}")
    print(f"Build report (MD): {BUILD_REPORT_MD}")
    print(f"Build report (JSON): {BUILD_REPORT_JSON}")


if __name__ == "__main__":
    main()