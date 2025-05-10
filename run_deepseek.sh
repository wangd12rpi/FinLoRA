#!/usr/bin/env bash
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-api=key}"
export DEEPSEEK_API_KEY
export OPENAI_BASE_URL="https://api.deepseek.com"

MODEL_NAME="deepseek-chat"
BATCH_SIZE=1
TEMPERATURE=0.0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PY="${REPO_ROOT}/test/test.py"
LOG_DIR="${REPO_ROOT}/test"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/deepseek_v3_results_${TS}.txt"

tasks=( financebench xbrl_term formula \
        xbrl_tags_extract xbrl_value_extract xbrl_formula_extract \
        xbrl_formula_calc_extract xbrl_finer xbrl_fnxl )

{
echo "============================================================"
echo "DeepSeek-V3 $(date)"
echo "============================================================"
for ds in "${tasks[@]}"; do
   echo "DATASET: $ds"
   python "$TEST_PY" --dataset "$ds" --source deepseek \
          --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE"
   echo
done
echo "============================================================"
} 2>&1 | tee "$LOG"
