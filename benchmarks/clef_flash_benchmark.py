#!/usr/bin/env python3
"""Independent LR vs local Ollama Clef Flash benchmark. Training is untouched."""

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL_DATA = ROOT / "outputs/universal_zh_classifier.source_aware.samples.jsonl"
PREVIEW_DATA = ROOT / "outputs/universal_zh_classifier.source_aware.preview.jsonl"
LR_MODEL = ROOT / "outputs/models/source_aware_multilabel_classifier.joblib"
FIXTURE = ROOT / "benchmarks/fixtures/clef_flash_dry_run.json"
REPORT = ROOT / "outputs/benchmarks/clef_flash_benchmark.json"
SAMPLES = ROOT / "outputs/benchmarks/clef_flash_samples.jsonl"
MIN_OLLAMA = (0, 35, 1)

LABELS = [
    "ACTION_OPERATION", "TOOL_SYSTEM", "ERROR_PROBLEM", "STATE_STATUS",
    "DATA_INFO", "GOAL_INTENT", "METHOD_STRATEGY", "RELATION_STRUCTURE",
    "ENTITY_OBJECT", "ATTRIBUTE_PROPERTY", "CONCEPT_ABSTRACT", "LANGUAGE_EXPRESSION",
]
DESCRIPTIONS = {
    "ACTION_OPERATION": "动作、操作、执行行为或过程",
    "TOOL_SYSTEM": "工具、软件、系统、平台、接口或基础设施",
    "ERROR_PROBLEM": "错误、故障、异常、失败、风险或待解决问题",
    "STATE_STATUS": "状态、阶段、条件、当前情况或运行状态",
    "DATA_INFO": "数据、信息、记录、内容、知识或可传递信息",
    "GOAL_INTENT": "目标、意图、目的、需求或期望结果",
    "METHOD_STRATEGY": "方法、策略、流程、方案、步骤或解决路径",
    "RELATION_STRUCTURE": "关系、结构、层级、连接、组成或组织方式",
    "ENTITY_OBJECT": "实体、对象、设备、人物、组织、物品或可指称事物",
    "ATTRIBUTE_PROPERTY": "属性、特征、参数、性质、质量或描述性维度",
    "CONCEPT_ABSTRACT": "抽象概念、理论、原则、思想、语义或认知概念",
    "LANGUAGE_EXPRESSION": "词语、句子、命名、表达方式、语言形式或释义",
}


def request_json(method, url, payload=None, timeout=180.0):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def parse_version(value):
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise RuntimeError(f"Unrecognized Ollama version: {value!r}")
    return tuple(map(int, match.groups()))


def sigmoid(value):
    value = max(min(float(value), 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-value))


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def rnd(value, digits=4):
    return None if value is None else round(value, digits)


def iter_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{n}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def sample_key(row):
    return str(row.get("source_record_id") or row.get("normalized_text") or row.get("text") or "")


def select_samples(path, limit, seed=7):
    """Prefer first examples that add new labels, then fill from a seeded reservoir."""
    if limit <= 0:
        raise RuntimeError("--limit must be > 0")
    selected, selected_keys, covered = [], set(), set()
    reservoir, seen = [], 0
    rng = random.Random(seed)
    reservoir_size = max(limit * 4, 64)
    label_set = set(LABELS)

    for row in iter_jsonl(path):
        text = str(row.get("normalized_text") or row.get("text") or "").strip()
        labels = set(row.get("cognitive_labels") or []) & label_set
        if not text or not labels:
            continue
        key = sample_key(row)
        if labels - covered and key not in selected_keys and len(selected) < limit:
            selected.append(row)
            selected_keys.add(key)
            covered.update(labels)
        seen += 1
        if len(reservoir) < reservoir_size:
            reservoir.append(row)
        else:
            idx = rng.randrange(seen)
            if idx < reservoir_size:
                reservoir[idx] = row

    for row in reservoir:
        if len(selected) >= limit:
            break
        key = sample_key(row)
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    return selected[:limit]


class Baseline:
    def __init__(self, path, threshold):
        import joblib
        started = time.perf_counter()
        bundle = joblib.load(path)
        self.load_ms = (time.perf_counter() - started) * 1000
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.names = list(bundle["label_binarizer"].classes_)
        self.threshold = threshold

    def predict(self, text):
        started = time.perf_counter()
        x = self.vectorizer.transform([text])
        scores = self.classifier.decision_function(x)
        if getattr(scores, "ndim", 1) == 1:
            scores = scores.reshape(1, -1)
        by_label = {name: sigmoid(scores[0][i]) for i, name in enumerate(self.names) if name in LABELS}
        latency = (time.perf_counter() - started) * 1000
        return {
            "labels": sorted(name for name, score in by_label.items() if score > self.threshold),
            "scores": {name: round(by_label.get(name, 0.0), 6) for name in LABELS},
            "latency_ms": latency,
        }


class Clef:
    def __init__(self, base_url, model, threshold, timeout, keep_alive):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.timeout = timeout
        self.keep_alive = keep_alive

    def validate(self):
        version = str(request_json("GET", self.base_url + "/api/version", timeout=self.timeout).get("version", ""))
        if parse_version(version) < MIN_OLLAMA:
            raise RuntimeError(f"Ollama {version} is too old; clef-flash requires >= 0.35.1")
        tags = request_json("GET", self.base_url + "/api/tags", timeout=self.timeout)
        installed = {str(m.get("name") or m.get("model") or "") for m in tags.get("models", []) if isinstance(m, dict)}
        base = self.model.split(":", 1)[0]
        if not any(name == self.model or name.split(":", 1)[0] == base for name in installed):
            raise RuntimeError(f"Model {self.model!r} is not installed. Run: ollama pull {self.model}. Or use --dry-run.")
        return {"version": version, "model": self.model}

    def questions(self):
        return {
            label: {
                "type": "noul",
                "instructions": f"判断输入文本是否应归入认知标签 {label}。这是多标签分类，每个标签独立判断；只根据文本语义判断。",
                "criteria": {
                    "true": f"文本明确表达或主要涉及：{DESCRIPTIONS[label]}。",
                    "false": f"文本不表达或不主要涉及：{DESCRIPTIONS[label]}。",
                },
            }
            for label in LABELS
        }

    def predict(self, text):
        payload = {"model": self.model, "state": text, "questions": self.questions(), "keep_alive": self.keep_alive}
        started = time.perf_counter()
        response = request_json("POST", self.base_url + "/v1/systemone", payload, self.timeout)
        latency = (time.perf_counter() - started) * 1000
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError(f"Unexpected clef-flash response: {response}")
        scores = {}
        for label in LABELS:
            answer = answers.get(label)
            if not isinstance(answer, dict) or "noul" not in answer:
                raise RuntimeError(f"Missing noul answer for {label}: {answer!r}")
            scores[label] = float(answer["noul"])
        return {
            "labels": sorted(label for label, score in scores.items() if score > self.threshold),
            "scores": {label: round(scores[label], 6) for label in LABELS},
            "latency_ms": latency,
            "usage": response.get("usage"),
        }


def class_metrics(rows, model_key, label):
    tp = fp = fn = tn = 0
    for row in rows:
        ref = label in row["reference_labels"]
        pred = label in row[model_key]["labels"]
        if ref and pred: tp += 1
        elif not ref and pred: fp += 1
        elif ref and not pred: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision_vs_weak_labels": rnd(precision), "recall_vs_weak_labels": rnd(recall), "f1_vs_weak_labels": rnd(f1)}


def summarize(rows, mode, setup=None):
    if not rows:
        raise RuntimeError("No benchmark rows")
    exact, jaccards = 0, []
    nonempty = Counter()
    weak_exact = Counter()
    unique = {"baseline": set(), "clef_flash": set()}
    latencies = {"baseline": [], "clef_flash": []}
    totals = {"baseline": Counter(), "clef_flash": Counter()}
    disagreement = Counter()

    for row in rows:
        ref = set(row["reference_labels"])
        b, c = set(row["baseline"]["labels"]), set(row["clef_flash"]["labels"])
        union = b | c
        exact += b == c
        jaccards.append(len(b & c) / len(union) if union else 1.0)
        for key, pred in (("baseline", b), ("clef_flash", c)):
            nonempty[key] += bool(pred)
            weak_exact[key] += pred == ref
            unique[key].update(pred)
            totals[key]["tp"] += len(pred & ref)
            totals[key]["fp"] += len(pred - ref)
            totals[key]["fn"] += len(ref - pred)
            if row[key].get("latency_ms") is not None:
                latencies[key].append(float(row[key]["latency_ms"]))
        for label in LABELS:
            if (label in b) != (label in c):
                disagreement[(label, "baseline_only" if label in b else "clef_only")] += 1

    model_summary = {}
    for key in ("baseline", "clef_flash"):
        t = totals[key]
        precision = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else None
        recall = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        model_summary[key] = {
            "sample_coverage": round(nonempty[key] / len(rows), 4),
            "taxonomy_coverage": round(len(unique[key]) / len(LABELS), 4),
            "predicted_labels": sorted(unique[key]),
            "exact_match_vs_weak_labels": round(weak_exact[key] / len(rows), 4),
            "micro_precision_vs_weak_labels": rnd(precision),
            "micro_recall_vs_weak_labels": rnd(recall),
            "micro_f1_vs_weak_labels": rnd(f1),
            "latency_ms": {
                "median": rnd(statistics.median(latencies[key]) if latencies[key] else None, 3),
                "mean": rnd(statistics.mean(latencies[key]) if latencies[key] else None, 3),
                "p95": rnd(percentile(latencies[key], 0.95), 3),
            },
        }

    per_class = {}
    for label in LABELS:
        agree = sum((label in row["baseline"]["labels"]) == (label in row["clef_flash"]["labels"]) for row in rows)
        per_class[label] = {
            "agreement_rate": round(agree / len(rows), 4),
            "baseline_only": disagreement[(label, "baseline_only")],
            "clef_only": disagreement[(label, "clef_only")],
            "baseline_vs_weak_labels": class_metrics(rows, "baseline", label),
            "clef_flash_vs_weak_labels": class_metrics(rows, "clef_flash", label),
        }

    return {
        "benchmark": "clef-flash-vs-lr",
        "mode": mode,
        "sample_count": len(rows),
        "reference_note": "cognitive_labels are weak-supervision references, not human gold labels",
        "setup": setup or {},
        "agreement": {"exact_label_set_rate": round(exact / len(rows), 4), "mean_jaccard": round(statistics.mean(jaccards), 4), "median_jaccard": round(statistics.median(jaccards), 4)},
        "coverage_and_weak_label_metrics": model_summary,
        "per_class": per_class,
        "samples": rows,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(report, out):
    print(f"mode: {report['mode']} | samples: {report['sample_count']}")
    print(f"LR <-> Clef exact agreement: {report['agreement']['exact_label_set_rate']:.1%}")
    print(f"LR <-> Clef mean Jaccard: {report['agreement']['mean_jaccard']:.3f}")
    for key, name in (("baseline", "LR"), ("clef_flash", "Clef Flash")):
        item = report["coverage_and_weak_label_metrics"][key]
        print(f"{name}: sample coverage={item['sample_coverage']:.1%}, taxonomy coverage={item['taxonomy_coverage']:.1%}, weak-label micro-F1={item['micro_f1_vs_weak_labels']}, median latency={item['latency_ms']['median']} ms")
    print(f"report: {out}")


def dry_run(fixture_path, out):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    report = summarize(fixture["samples"], "dry-run", {"fixture": str(fixture_path), "synthetic": True})
    write_json(out, report)
    print_summary(report, out)
    print("note: dry-run numbers are synthetic; they validate plumbing only")
    return 0


def live(args):
    data = Path(args.data) if args.data else (FULL_DATA if FULL_DATA.exists() else PREVIEW_DATA)
    lr_path = Path(args.baseline_model)
    if not data.exists():
        raise RuntimeError(f"Dataset not found: {data}")
    if not lr_path.exists():
        raise RuntimeError(f"Baseline model not found: {lr_path}. Run python src/train_baseline.py first, or use --dry-run.")

    samples = select_samples(data, args.limit, args.seed)
    if not samples:
        raise RuntimeError(f"No usable samples found in {data}")
    write_jsonl(Path(args.sample_out), samples)

    baseline = Baseline(lr_path, args.threshold)
    clef = Clef(args.ollama_url, args.clef_model, args.threshold, args.timeout, args.keep_alive)
    ollama = clef.validate()
    cold_start_ms = None
    for i in range(args.warmup):
        text = str(samples[i % len(samples)].get("normalized_text") or samples[i % len(samples)]["text"])
        started = time.perf_counter()
        clef.predict(text)
        if i == 0:
            cold_start_ms = (time.perf_counter() - started) * 1000

    rows = []
    for i, sample in enumerate(samples, 1):
        text = str(sample.get("normalized_text") or sample.get("text") or "").strip()
        reference = sorted(set(sample.get("cognitive_labels") or []) & set(LABELS))
        rows.append({
            "index": i, "text": text, "source_record_id": sample.get("source_record_id"),
            "reference_labels": reference, "baseline": baseline.predict(text), "clef_flash": clef.predict(text),
        })
        print(f"[{i}/{len(samples)}] {text[:48]!r}")

    setup = {
        "data": str(data), "baseline_model": str(lr_path), "baseline_load_ms": round(baseline.load_ms, 3),
        "threshold": args.threshold, "ollama": ollama, "clef_cold_start_ms": rnd(cold_start_ms, 3),
        "warmup_requests": args.warmup, "keep_alive": args.keep_alive,
    }
    report = summarize(rows, "live", setup)
    out = Path(args.out)
    write_json(out, report)
    print_summary(report, out)
    return 0


def parser():
    p = argparse.ArgumentParser(description="Benchmark current LR baseline against local Ollama clef-flash")
    p.add_argument("--dry-run", action="store_true", help="Use synthetic fixture; no model/Ollama required")
    p.add_argument("--fixture", default=str(FIXTURE))
    p.add_argument("--data")
    p.add_argument("--baseline-model", default=str(LR_MODEL))
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--clef-model", default="clef-flash")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--keep-alive", default="10m")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--sample-out", default=str(SAMPLES))
    p.add_argument("--out", default=str(REPORT))
    return p


def main():
    args = parser().parse_args()
    if not 0.0 < args.threshold < 1.0:
        print("error: --threshold must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        return dry_run(Path(args.fixture), Path(args.out)) if args.dry_run else live(args)
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
