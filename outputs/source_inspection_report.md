# Source Inspection Report

Generated: 2026-07-04 13:18:48

## 1. Overview

This report inspects candidate GitHub open-source Chinese dictionary / lexicon
datasets for building the **source-aware JSONL classifier dataset**.

| Source | Role | License | Local Files | Status |
|--------|------|---------|-------------|--------|
| pwxcoo/chinese-xinhua | primary_bootstrap_source | Repo shows MIT license, but README says dictionary data was ... | 0 | not_fetched |
| g0v/moedict-data | supplemental_source | Community-driven derivative of Taiwan Ministry of Education ... | 0 | not_fetched |
| thunlp/THUOCL | candidate_supplemental_source | Academic domain lexicon dataset from Tsinghua University. Ty... | 0 | not_fetched |
| liuhuanyong/DomainWordsDict | candidate_supplemental_source | Domain-specific Chinese word lists curated by academic resea... | 0 | not_fetched |

## 2. Per-Source Analysis

### pwxcoo/chinese-xinhua

- **GitHub**: https://github.com/pwxcoo/chinese-xinhua
- **Recommended role**: primary_bootstrap_source
- **License hint**: Repo shows MIT license, but README says dictionary data was collected/scraped from the web. Suitable for research/prototyping; verify before commercial redistribution.
- **Language scope**: Simplified Chinese; words, idioms, characters, xiehouyu

**Raw URLs**:

- `ci`: https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/ci.json
- `idiom`: https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/idiom.json
- `word`: https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/word.json
- `xiehouyu`: https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/xiehouyu.json

**Expected fields**:

- `ci`: ci, explanation
- `idiom`: word, pinyin, abbreviation, explanation, derivation, example
- `word`: word, oldword, strokes, pinyin, radicals, explanation, more
- `xiehouyu`: riddle, answer

> No local files found. Run `python src/fetch_sources.py` first for primary/supplemental sources.

**README excerpt** (first 500 chars):
```
# chinese-xinhua

中华新华字典数据库和 API 。收录包括 14032 条歇后语，16142 个汉字，264434 个词语，31648 个成语。

## Project Structure

```
chinese-xinhua/
|
+- data/ <-- 数据文件夹
|  |
|  +- idiom.json <-- 成语
|  |
|  +- word.json <-- 汉字
|  |
|  +- xiehouyu.json <-- 歇后语
|  |
|  +- ci.json <-- 词语
```

## Database Introduction

### 成语 (idiom.json)

```json
[
    {
        "derivation": "语出《法华经·法师功德品》下至阿鼻地狱。”",
        "example": "但也有少数意志薄弱的……逐步上当，终至堕入～。★《上饶集中营·炼狱杂记》",
        "explanation": "阿鼻梵语的译音，意译为无间”，即痛苦无有间断之意。常用来比喻黑暗的社会和严酷的牢
```

**Assessment**:

- Suitable as **primary bootstrap source** for the simplified Chinese classifier dataset.
- Large vocabulary coverage across words, idioms, characters, and xiehouyu.
- Data originates from web-scraped sources; full redistribution and commercial use rights need verification.
- Fields are well-structured JSON; suitable for automated processing and weak supervision.

- **Suitable for direct training?** This source can seed weakly-supervised labels but **is not human gold-standard**; samples should be reviewed before deployment.
- **Commercial / redistribution risk**: Moderate — data is scraped / academically sourced; verify before redistribution.

### g0v/moedict-data

- **GitHub**: https://github.com/g0v/moedict-data
- **Recommended role**: supplemental_source
- **License hint**: Community-driven derivative of Taiwan Ministry of Education dictionary data. Contains traditional Chinese definitions. CC0 / public domain dedication for the dataset structure; original dictionary text may have usage restrictions. Use as a supplemental source; do not redistribute raw text without verification.
- **Language scope**: Traditional / Taiwan Ministry of Education Mandarin dictionary; rich multi-sense definitions.

**Raw URLs**:

- `dict-revised`: https://raw.githubusercontent.com/g0v/moedict-data/master/dict-revised.json

**Expected fields**:

- `dict-revised`: title, heteronyms, radical, non_radical_stroke_count, total_stroke, pinyin

> No local files found. Run `python src/fetch_sources.py` first for primary/supplemental sources.

**README excerpt** (first 500 chars):
```
這是將「重編國語辭典（修訂本）」的公眾授權內容處理為機器比較容易再利用的 json 格式。辭典本文的著作權仍為教育部所有。

公眾授權網：https://language.moe.gov.tw/001/Upload/Files/site_content/M0001/respub/index.html

依教育部之解釋，「創用CC-姓名標示- 禁止改作 臺灣3.0版授權條款」之改作限制標的為文字資料本身，不限制格式轉換及後續應用。

	=====================================================
	企劃執行：國家教育研究院
	
	原 著 者：教育部國語推行委員會
	　　　   （民國102年1月1日配合行政院組改併入相關單位）
	
	發 行 人：潘文忠　林崇熙
	
	發 行 所：中華民國教育部
	
	維護單位：國家教育研究院語文教育及編譯研究中心
	
	地　　址：臺北市大安區和平東路一段179號
	
	電　　話：(02)7740-7282
	
	傳　　真：(02)7740-7284
	
	電子郵件：onile@mail.naer.edu.tw
```

**Assessment**:

- Suitable as a **supplemental source** to enrich definitions and multi-sense entries.
- Traditional / Taiwan Mandarin dictionary system; should not overwrite simplified primary vocab.
- Use to cross-reference and supplement, not replace primary source labels.
- License restrictions on original dictionary text require verification before redistribution.

- **Suitable for direct training?** This source can seed weakly-supervised labels but **is not human gold-standard**; samples should be reviewed before deployment.
- **Commercial / redistribution risk**: Review terms carefully before any commercial use.

### thunlp/THUOCL

- **GitHub**: https://github.com/thunlp/THUOCL
- **Recommended role**: candidate_supplemental_source
- **License hint**: Academic domain lexicon dataset from Tsinghua University. Typically used for research/education; verify licensing terms before commercial use. Chinese Thematic Word List organized by domains.
- **Language scope**: Simplified / Traditional Chinese; domain-specific word lists (medical, financial, IT, food, etc.)

> No local files found. Run `python src/fetch_sources.py` first for primary/supplemental sources.

**README excerpt** (first 500 chars):
```
# THUOCL

## 目录

* [词库简介](#词库简介)
* [词库格式及词频统计语料库](#词库格式及词频统计语料库)
* [词库清单](#词库清单)
* [开源协议](#开源协议)
* [作者](#作者)

## 词库介绍

THUOCL（THU Open Chinese Lexicon）是由清华大学自然语言处理与社会人文计算实验室整理推出的一套高质量的中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。THUOCL具有以下特点：

1. 包含词频统计信息DF值（Document Frequency），方便用户个性化选择使用。

2. 词库经过多轮人工筛选，保证词库收录的准确性。

3. 开放更新，将不断更新现有词表，并推出更多类别词表。欢迎专业人士加入，协作建设开放词库，有意者请致信thunlp@gmail.com。

该词库可以用于中文自动分词，提升中文分词效果。建议搭配本组研制开发的[THULAC工具包](http://thulac.thunlp.org/)使用，提升特定领域中文分词的效果。

## 词库格式及词频统计语料库

词库每一行由两部分组成，分别是
```

**Assessment**:

- **Candidate supplemental source**: potentially useful for domain-specific enrichment.
- Not pulled in by default; would require download-and-parse integration.
- Determine fitness after initial dataset is built and domain gaps are identified.

- **Suitable for direct training?** This source can seed weakly-supervised labels but **is not human gold-standard**; samples should be reviewed before deployment.
- **Commercial / redistribution risk**: Moderate — data is scraped / academically sourced; verify before redistribution.

### liuhuanyong/DomainWordsDict

- **GitHub**: https://github.com/liuhuanyong/DomainWordsDict
- **Recommended role**: candidate_supplemental_source
- **License hint**: Domain-specific Chinese word lists curated by academic researcher. Useful for domain label enrichment; verify license before commercial use.
- **Language scope**: Simplified Chinese; domain word dictionaries for IT, medical, financial, automotive, etc.

> No local files found. Run `python src/fetch_sources.py` first for primary/supplemental sources.

> README fetch failed: master: HTTP Error 404: Not Found / main: HTTP Error 404: Not Found

**Assessment**:

- **Candidate supplemental source**: potentially useful for domain-specific enrichment.
- Not pulled in by default; would require download-and-parse integration.
- Determine fitness after initial dataset is built and domain gaps are identified.

- **Suitable for direct training?** This source can seed weakly-supervised labels but **is not human gold-standard**; samples should be reviewed before deployment.
- **Commercial / redistribution risk**: Moderate — data is scraped / academically sourced; verify before redistribution.

## 3. Summary & Recommendations

1. **pwxcoo/chinese-xinhua** is chosen as the primary bootstrap source because:
   - Clean, structured JSON covering words, idioms, characters, xiehouyu.
   - Simplified Chinese, directly applicable to the target classifier.
   - MIT-licensed repo (though data provenance requires verification).

2. **g0v/moedict-data** is recommended as a supplemental source because:
   - Rich multi-sense definitions from Taiwan Ministry of Education dictionary.
   - Complements simplified source with Traditional Chinese definitions.
   - Structure may be complex; parsed into simple records for supplement.

3. **THUOCL** and **DomainWordsDict** are candidate sources for future domain enrichment,
   not fetched by default but recorded in the catalog for later inclusion.

4. **All data is weak-supervision / research-prototyping grade**.
   Samples generated with `source_default_labels + keyword_weak_supervision` are **not**
   human-verified gold labels; `needs_review` flags low-confidence entries.
