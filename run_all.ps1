# run_all.ps1 — End-to-end pipeline for Windows PowerShell
# Usage: .\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "============================================================"
Write-Host "Step 1: Inspect sources"
Write-Host "============================================================"
python src/inspect_sources.py

Write-Host ""
Write-Host "============================================================"
Write-Host "Step 2: Fetch raw sources"
Write-Host "============================================================"
python src/fetch_sources.py

Write-Host ""
Write-Host "============================================================"
Write-Host "Step 3: Build source-aware JSONL dataset"
Write-Host "============================================================"
python src/build_dataset.py

Write-Host ""
Write-Host "============================================================"
Write-Host "Step 4: Train baseline classifier"
Write-Host "============================================================"
python src/train_baseline.py

Write-Host ""
Write-Host "============================================================"
Write-Host "Step 5: Run prediction examples"
Write-Host "============================================================"
Write-Host "--- Test 1: 接口调用失败 ---"
python src/predict.py --text "接口调用失败"
Write-Host ""
Write-Host "--- Test 2: Agent 调用工具失败 ---"
python src/predict.py --text "Agent 调用工具失败"
Write-Host ""
Write-Host "--- Test 3: 机器人路径规划 ---"
python src/predict.py --text "机器人路径规划"

Write-Host ""
Write-Host "============================================================"
Write-Host "All steps complete."
Write-Host "============================================================"