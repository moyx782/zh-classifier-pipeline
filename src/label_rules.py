#!/usr/bin/env python3
"""label_rules.py — Weak-supervision keyword rules for cognitive and domain labels.

This module provides keyword-based weak supervision to augment source-default labels.
For each cognitive/domain label, keyword matches are performed on the text plus
raw_fields (definition / explanation / example / answer / more, etc.).

Usage:
    from label_rules import apply_weak_supervision
"""

COGNITIVE_RULES = {
    "ACTION_OPERATION": [
        "调用", "执行", "运行", "读取", "写入", "删除", "修改", "搜索",
        "查询", "分类", "聚类", "训练", "推理", "部署", "生成", "转换",
        "解析", "识别", "连接", "同步", "检索", "抽取", "构建",
    ],
    "TOOL_SYSTEM": [
        "工具", "系统", "平台", "接口", "API", "SDK", "框架", "插件",
        "服务", "软件", "模型", "服务器", "客户端", "数据库", "程序",
        "应用", "引擎", "组件", "模块",
    ],
    "ERROR_PROBLEM": [
        "失败", "错误", "异常", "报错", "冲突", "超时", "不可用", "缺失",
        "中断", "崩溃", "无法", "问题", "障碍",
    ],
    "STATE_STATUS": [
        "成功", "失败", "运行中", "空闲", "阻塞", "可用", "不可用", "稳定",
        "活跃", "停用",
    ],
    "DATA_INFO": [
        "文件", "参数", "配置", "日志", "上下文", "数据", "元数据", "字段",
        "词条", "记录", "数据库", "样本", "语料", "释义", "拼音",
    ],
    "GOAL_INTENT": [
        "目标", "需求", "意图", "任务", "计划", "目的", "希望", "需要",
        "用于", "为了",
    ],
    "METHOD_STRATEGY": [
        "方法", "策略", "方案", "流程", "规则", "算法", "协议", "路径",
        "规划", "模板", "规范", "标准",
    ],
    "RELATION_STRUCTURE": [
        "连接", "映射", "依赖", "组合", "层级", "结构", "关系", "接口",
        "链路", "父", "子", "对应",
    ],
    "ENTITY_OBJECT": [
        "机器人", "无人机", "用户", "设备", "组件", "模块", "对象", "实体",
        "节点", "字", "词", "成语", "歇后语",
    ],
    "ATTRIBUTE_PROPERTY": [
        "大小", "速度", "权重", "置信度", "频率", "笔画", "部首", "拼音",
        "参数", "属性", "特征",
    ],
    "CONCEPT_ABSTRACT": [
        "思想", "语义", "认知", "概念", "理论", "分类", "聚类", "模式",
        "表征", "意义", "定义", "抽象",
    ],
    "LANGUAGE_EXPRESSION": [
        "词", "句", "释义", "拼音", "成语", "歇后语", "同义词", "例句",
        "部首", "笔画", "谜面", "答案", "字", "汉字",
    ],
}

DOMAIN_RULES = {
    "SOFTWARE": [
        "接口", "API", "SDK", "框架", "插件", "服务", "软件", "模型",
        "服务器", "客户端", "数据库", "调用", "部署", "程序", "应用",
    ],
    "ML": [
        "模型", "训练", "推理", "分类", "聚类", "语义", "表征", "置信度",
        "Transformer", "embedding", "LoRA", "RAG", "样本",
    ],
    "ROBOTICS": [
        "机器人", "无人机", "ROS", "ROS2", "SLAM", "导航", "路径规划",
        "雷达", "机械臂", "底盘",
    ],
    "AGENT_TOOLING": [
        "Agent", "MCP", "ACP", "tool", "工具调用", "schema", "workflow",
        "session", "上下文", "路由",
    ],
    "NETWORK": [
        "连接", "链路", "超时", "服务器", "客户端", "同步", "DNS", "代理",
        "路由", "OpenClash", "分流",
    ],
    "CHINESE_DICTIONARY": [
        "词", "句", "释义", "拼音", "成语", "歇后语", "部首", "笔画",
        "谜面", "答案", "字", "汉字",
    ],
    "GENERAL": [],
}


def _matches(text, keywords):
    """Return the list of keywords found in text."""
    hits = []
    for kw in keywords:
        if kw and kw in text:
            hits.append(kw)
    return hits


def _concatenate_for_matching(text, raw_fields):
    """Concatenate text + relevant raw_fields for keyword matching."""
    parts = [text or ""]
    if raw_fields:
        for key in [
            "definition", "explanation", "example", "answer", "more",
            "derivation", "pinyin", "abbreviation", "riddle",
            "oldword", "strokes", "radicals", "ci",
        ]:
            val = raw_fields.get(key, "")
            if isinstance(val, str):
                parts.append(val)
            else:
                parts.append(str(val))
    return " ".join(parts)


def apply_weak_supervision(text, raw_fields, base_cognitive=None, base_domain=None,
                           base_confidence=0.5, max_confidence=0.95,
                           keyword_bonus=0.04):
    """Apply weak-supervision keyword rules.

    Returns (cognitive_labels, domain_labels, confidence, label_evidence, labeling_method).

    Parameters
    ----------
    text : str
        The primary text being labeled.
    raw_fields : dict
        Original raw fields from the source record, used for extended matching.
    base_cognitive : list[str]
        Cognitive labels already assigned by source default rules.
    base_domain : list[str]
        Domain labels already assigned by source default rules.
    base_confidence : float
        Starting confidence for source-default-labeled samples.
    max_confidence : float
        Confidence cap.

    Returns
    -------
    dict with keys:
        cognitive_labels, domain_labels, confidence, label_evidence, labeling_method
    """
    base_cognitive = list(base_cognitive or [])
    base_domain = list(base_domain or [])

    text_all = _concatenate_for_matching(text, raw_fields)

    cognitive_labels = set(base_cognitive)
    domain_labels = set(base_domain)
    label_evidence = {}

    # Cognitive rules
    for label, keywords in COGNITIVE_RULES.items():
        hits = _matches(text_all, keywords)
        if hits:
            if label not in cognitive_labels:
                cognitive_labels.add(label)
            label_evidence.setdefault(label, []).extend(hits)

    # Domain rules
    for label, keywords in DOMAIN_RULES.items():
        if not keywords:
            continue
        hits = _matches(text_all, keywords)
        if hits:
            if label not in domain_labels:
                domain_labels.add(label)
            label_evidence.setdefault(label, []).extend(hits)

    # Always include GENERAL if no domain hit at all
    if not domain_labels:
        domain_labels.add("GENERAL")
    # Always can fall to GENERAL
    if "GENERAL" not in domain_labels and len(domain_labels) == 0:
        domain_labels.add("GENERAL")

    # Confidence: start from base, add bonus per matched label (capped)
    total_hits = sum(len(v) for v in label_evidence.values())
    confidence = base_confidence + min(total_hits, 10) * keyword_bonus
    confidence = min(confidence, max_confidence)

    # Labeling method
    has_kw = total_hits > 0
    if has_kw and base_cognitive:
        labeling_method = "source_default_labels + keyword_weak_supervision"
    elif has_kw:
        labeling_method = "keyword_weak_supervision"
    else:
        labeling_method = "source_default_labels"

    return {
        "cognitive_labels": sorted(cognitive_labels),
        "domain_labels": sorted(domain_labels),
        "confidence": round(confidence, 4),
        "label_evidence": label_evidence,
        "labeling_method": labeling_method,
    }


if __name__ == "__main__":
    sample = apply_weak_supervision(
        "接口调用失败",
        {"explanation": "接口调用失败时需要排查连接问题。"},
        base_cognitive=["LANGUAGE_EXPRESSION", "DATA_INFO"],
        base_domain=["CHINESE_DICTIONARY"],
    )
    import json
    print(json.dumps(sample, ensure_ascii=False, indent=2))