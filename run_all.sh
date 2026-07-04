#!/usr/bin/env bash
# run_all.sh — End-to-end pipeline
# Usage: bash run_all.sh

set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "Step 1: Inspect sources"
echo "============================================================"
python src/inspect_sources.py

echo ""
echo "============================================================"
echo "Step 2: Fetch raw sources"
echo "============================================================"
python src/fetch_sources.py

echo ""
echo "============================================================"
echo "Step 3: Build source-aware JSONL dataset"
echo "============================================================"
python src/build_dataset.py

echo ""
echo "============================================================"
echo "Step 4: Train baseline classifier"
echo "============================================================"
python src/train_baseline.py

echo ""
echo "============================================================"
echo "Step 5: Run prediction examples"
echo "============================================================"
echo "--- Test 1: 接口调用失败 ---"
python src/predict.py --text "接口调用失败"
echo ""
echo "--- Test 2: Agent 调用工具失败 ---"
python src/predict.py --text "Agent 调用工具失败"
echo ""
echo "--- Test 3: 机器人路径规划 ---"
python src/predict.py --text "机器人路径规划"

echo ""
echo "============================================================"
echo "All steps complete."
echo "============================================================"