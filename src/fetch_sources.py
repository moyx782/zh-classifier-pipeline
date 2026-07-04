#!/usr/bin/env python3
"""fetch_sources.py — Download raw data from GitHub open-source Chinese dictionary projects.

Primary source: pwxcoo/chinese-xinhua
Supplemental source: g0v/moedict-data

All fetched files are stored under raw/<repo_key>/.
A fetch_report.json is written to outputs/ with download status, record counts,
and field structure summaries.

Usage:
    python src/fetch_sources.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CATALOG_PATH = OUTPUTS_DIR / "source_catalog.json"
FETCH_REPORT_PATH = OUTPUTS_DIR / "fetch_report.json"

# ---- Helpers --------------------------------------------------------------

def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def download(url, dest_path, timeout=60, retries=3):
    """Download a file from url to dest_path with retries. Returns (success, size_bytes, error_msg)."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research-fetcher)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            return True, len(data), None
        except Exception as e:
            err = str(e)
            if attempt < retries:
                time.sleep(2 * attempt)
            continue
    return False, 0, err


def summarize_fields(records, max_keys=20):
    """Aggregate field keys across the first N records."""
    if not isinstance(records, list) or not records:
        return {}
    key_count = {}
    for rec in records[:5000]:
        if isinstance(rec, dict):
            for k in rec.keys():
                key_count[k] = key_count.get(k, 0) + 1
    # Sort by frequency
    sorted_keys = sorted(key_count.items(), key=lambda x: -x[1])
    return {k: c for k, c in sorted_keys[:max_keys]}


# ---- Main fetch -----------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    fetch_report = {
        "fetch_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {},
    }

    # Directory mapping for known repos
    raw_subdirs = {
        "pwxcoo/chinese-xinhua": RAW_DIR / "pwxcoo__chinese-xinhua",
        "g0v/moedict-data": RAW_DIR / "g0v__moedict-data",
    }

    for repo_key, info in catalog.items():
        raw_urls = info.get("raw_urls", {})
        if not raw_urls:
            fetch_report["sources"][repo_key] = {
                "status": "no_raw_urls",
                "license_hint": info.get("license_hint", ""),
                "recommended_role": info.get("recommended_role", ""),
                "files": {},
            }
            continue

        raw_subdir = raw_subdirs.get(repo_key, RAW_DIR / repo_key.replace("/", "__"))
        raw_subdir.mkdir(parents=True, exist_ok=True)

        source_report = {
            "status": "pending",
            "license_hint": info.get("license_hint", ""),
            "recommended_role": info.get("recommended_role", ""),
            "files": {},
        }

        for dataset_name, url in raw_urls.items():
            filename = url.split("/")[-1]
            dest = raw_subdir / filename
            ok, size, err = download(url, dest)
            file_info = {
                "url": url,
                "local_path": str(dest.relative_to(PROJECT_ROOT)),
                "downloaded": ok,
                "size_bytes": size,
                "record_count": 0,
                "field_summary": {},
            }
            if ok:
                try:
                    with open(dest, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        file_info["record_count"] = len(data)
                        file_info["field_summary"] = summarize_fields(data)
                    elif isinstance(data, dict):
                        file_info["record_count"] = len(data)
                        file_info["field_summary"] = summarize_fields(
                            list(data.values())[:1000]
                        )
                    file_info["error"] = None
                except Exception as e:
                    file_info["error"] = f"JSON parse error: {e}"
            else:
                file_info["error"] = err

            source_report["files"][dataset_name] = file_info

        all_ok = all(
            fi["downloaded"] and "error" not in fi
            for fi in source_report["files"].values()
        ) if source_report["files"] else False
        source_report["status"] = "success" if all_ok else "partial_or_failed"

        fetch_report["sources"][repo_key] = source_report

    with open(FETCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(fetch_report, f, ensure_ascii=False, indent=2)

    # Print summary
    print("=" * 60)
    print("fetch_sources.py — Fetch Report")
    print("=" * 60)
    total_files = 0
    total_records = 0
    for repo_key, src in fetch_report["sources"].items():
        print(f"\n[{repo_key}]  role={src['recommended_role']}  status={src['status']}")
        for ds, fi in src.get("files", {}).items():
            total_files += 1
            total_records += fi.get("record_count", 0)
            print(f"  {ds}: downloaded={fi['downloaded']}  records={fi.get('record_count',0)}  size={fi.get('size_bytes',0)}B")
            if fi.get("error"):
                print(f"    ERROR: {fi['error']}")
    print(f"\nTotal files fetched: {total_files}")
    print(f"Total records: {total_records}")
    print(f"Report saved to: {FETCH_REPORT_PATH}")


if __name__ == "__main__":
    main()