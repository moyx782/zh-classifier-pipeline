#!/usr/bin/env python3
"""inspect_sources.py — Inspect GitHub data sources; generate source_inspection_report.md.

Checks:
- Catalog the known GitHub Chinese dictionary datasets.
- Determine if raw files are available locally (already fetched).
- Attempt to fetch metadata / README from each repo via GitHub raw URLs.
- Summarise license, scope, fields, quality risks, commercial redistribution risks.

Output: outputs/source_inspection_report.md

Usage:
    python src/inspect_sources.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = PROJECT_ROOT / "raw"
CATALOG_PATH = OUTPUTS_DIR / "source_catalog.json"
FETCH_REPORT_PATH = OUTPUTS_DIR / "fetch_report.json"
REPORT_PATH = OUTPUTS_DIR / "source_inspection_report.md"


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fetch_report():
    if FETCH_REPORT_PATH.exists():
        with open(FETCH_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def try_fetch_readme(repo_key, timeout=20):
    """Try to fetch README.md from repo."""
    url = f"https://raw.githubusercontent.com/{repo_key}/master/README.md"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-inspector)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace"), None
    except Exception as e:
        # Try main branch
        url2 = f"https://raw.githubusercontent.com/{repo_key}/main/README.md"
        try:
            req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                return resp2.read().decode("utf-8", errors="replace"), None
        except Exception as e2:
            return None, f"master: {e} / main: {e2}"


def inspect_local_files(repo_key, fetch_report):
    """Inspect locally available files for a repo."""
    subdir_name = repo_key.replace("/", "__")
    subdir = RAW_DIR / subdir_name
    info = {}
    if subdir.exists():
        json_files = list(subdir.glob("*.json"))
        info["local_files"] = [f.name for f in json_files]
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    info[jf.name] = {"records": len(data), "type": "list"}
                    if data:
                        info[jf.name]["sample_fields"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                else:
                    info[jf.name] = {"records": len(data), "type": "dict"}
            except Exception as e:
                info[jf.name] = {"error": str(e)}
    else:
        info["local_files"] = []
    return info


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    fetch_report = load_fetch_report()

    lines = []
    lines.append("# Source Inspection Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This report inspects candidate GitHub open-source Chinese dictionary / lexicon")
    lines.append("datasets for building the **source-aware JSONL classifier dataset**.")
    lines.append("")
    lines.append("| Source | Role | License | Local Files | Status |")
    lines.append("|--------|------|---------|-------------|--------|")
    for repo_key, info in catalog.items():
        role = info.get("recommended_role", "")
        license_short = info.get("license_hint", "")[:60] + "..."
        local_inspect = inspect_local_files(repo_key, fetch_report)
        n_files = len(local_inspect.get("local_files", []))
        fetch_status = fetch_report.get("sources", {}).get(repo_key, {}).get("status", "not_fetched")
        lines.append(f"| {repo_key} | {role} | {license_short} | {n_files} | {fetch_status} |")
    lines.append("")

    lines.append("## 2. Per-Source Analysis")
    lines.append("")

    for repo_key, info in catalog.items():
        lines.append(f"### {repo_key}")
        lines.append("")
        lines.append(f"- **GitHub**: {info.get('source_repo', '')}")
        lines.append(f"- **Recommended role**: {info.get('recommended_role', '')}")
        lines.append(f"- **License hint**: {info.get('license_hint', '')}")
        lines.append(f"- **Language scope**: {info.get('language_scope', '')}")
        lines.append("")

        # Raw URLs
        raw_urls = info.get("raw_urls", {})
        if raw_urls:
            lines.append("**Raw URLs**:")
            lines.append("")
            for ds, url in raw_urls.items():
                lines.append(f"- `{ds}`: {url}")
            lines.append("")

        # Expected fields
        expected = info.get("expected_fields", {})
        if expected:
            lines.append("**Expected fields**:")
            lines.append("")
            for ds, fields in expected.items():
                lines.append(f"- `{ds}`: {', '.join(fields)}")
            lines.append("")

        # Local inspection
        local_inspect = inspect_local_files(repo_key, fetch_report)
        local_files = local_inspect.get("local_files", [])
        if local_files:
            lines.append("**Local files available**:")
            lines.append("")
            for lf in local_files:
                fi = local_inspect.get(lf, {})
                n_records = fi.get("records", "?")
                sfields = fi.get("sample_fields", [])
                lines.append(f"- `{lf}`: {n_records} records; sample fields: {', '.join(sfields[:8])}")
            lines.append("")
        else:
            lines.append("> No local files found. Run `python src/fetch_sources.py` first for primary/supplemental sources.")
            lines.append("")

        # Fetch report details
        fr_entry = fetch_report.get("sources", {}).get(repo_key, {})
        if fr_entry:
            lines.append("**Fetch report**:")
            lines.append("")
            for ds, fi in fr_entry.get("files", {}).items():
                ok = fi.get("downloaded", False)
                n = fi.get("record_count", 0)
                err = fi.get("error")
                status = "OK" if ok and not err else "FAILED"
                lines.append(f"- `{ds}`: {status} (records={n})")
                if err:
                    lines.append(f"  - Error: {err}")
            overall = fr_entry.get("status", "")
            lines.append(f"- Overall status: {overall}")
            lines.append("")

        # README fetch
        readme_content, readme_err = try_fetch_readme(repo_key)
        if readme_content:
            lines.append("**README excerpt** (first 500 chars):")
            lines.append("```")
            lines.append(readme_content[:500])
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"> README fetch failed: {readme_err}")
            lines.append("")

        # Assessment
        lines.append("**Assessment**:")
        lines.append("")
        role = info.get("recommended_role", "")
        if role == "primary_bootstrap_source":
            lines.append("- Suitable as **primary bootstrap source** for the simplified Chinese classifier dataset.")
            lines.append("- Large vocabulary coverage across words, idioms, characters, and xiehouyu.")
            lines.append("- Data originates from web-scraped sources; full redistribution and commercial"
                        " use rights need verification.")
            lines.append("- Fields are well-structured JSON; suitable for automated processing and weak supervision.")
        elif role == "supplemental_source":
            lines.append("- Suitable as a **supplemental source** to enrich definitions and multi-sense entries.")
            lines.append("- Traditional / Taiwan Mandarin dictionary system; should not overwrite simplified primary vocab.")
            lines.append("- Use to cross-reference and supplement, not replace primary source labels.")
            lines.append("- License restrictions on original dictionary text require verification before redistribution.")
        elif "candidate" in role:
            lines.append("- **Candidate supplemental source**: potentially useful for domain-specific enrichment.")
            lines.append("- Not pulled in by default; would require download-and-parse integration.")
            lines.append("- Determine fitness after initial dataset is built and domain gaps are identified.")
        lines.append("")
        lines.append("- **Suitable for direct training?** This source can seed weakly-supervised labels but **is "
                     "not human gold-standard**; samples should be reviewed before deployment.")
        lines.append("- **Commercial / redistribution risk**: " + (
            "Moderate — data is scraped / academically sourced; verify before redistribution."
            if "verify" in info.get("license_hint", "").lower()
            else "Review terms carefully before any commercial use."
        ))
        lines.append("")

    lines.append("## 3. Summary & Recommendations")
    lines.append("")
    lines.append("1. **pwxcoo/chinese-xinhua** is chosen as the primary bootstrap source because:")
    lines.append("   - Clean, structured JSON covering words, idioms, characters, xiehouyu.")
    lines.append("   - Simplified Chinese, directly applicable to the target classifier.")
    lines.append("   - MIT-licensed repo (though data provenance requires verification).")
    lines.append("")
    lines.append("2. **g0v/moedict-data** is recommended as a supplemental source because:")
    lines.append("   - Rich multi-sense definitions from Taiwan Ministry of Education dictionary.")
    lines.append("   - Complements simplified source with Traditional Chinese definitions.")
    lines.append("   - Structure may be complex; parsed into simple records for supplement.")
    lines.append("")
    lines.append("3. **THUOCL** and **DomainWordsDict** are candidate sources for future domain enrichment,")
    lines.append("   not fetched by default but recorded in the catalog for later inclusion.")
    lines.append("")
    lines.append("4. **All data is weak-supervision / research-prototyping grade**.")
    lines.append("   Samples generated with `source_default_labels + keyword_weak_supervision` are **not**")
    lines.append("   human-verified gold labels; `needs_review` flags low-confidence entries.")
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 60)
    print("inspect_sources.py — Inspection complete")
    print("=" * 60)
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Sources inspected: {len(catalog)}")
    for repo_key in catalog:
        local_inspect = inspect_local_files(repo_key, fetch_report)
        n = len(local_inspect.get("local_files", []))
        print(f"  {repo_key}: {n} local file(s)")


if __name__ == "__main__":
    main()