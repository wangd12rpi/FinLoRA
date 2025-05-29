#!/usr/bin/env bash
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-proj-api_key}"
export OPENAI_API_KEY

MODEL_NAME="gpt-4o"
BATCH_SIZE=1
TEMPERATURE=0.0

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_PY="${REPO_ROOT}/test.py"
LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/gpt4o_results_${TS}.txt"

tasks=( financebench xbrl_term formula \
        xbrl_tags_extract xbrl_value_extract xbrl_formula_extract \
        xbrl_formula_calc_extract xbrl_finer xbrl_fnxl )

{
echo "============================================================"
echo "GPT‑4o $(date)"
echo "============================================================"
for ds in "${tasks[@]}"; do
   echo "DATASET: $ds"
   python "$TEST_PY" --dataset "$ds" --source openai \
          --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE"
   echo
done
echo "============================================================"
} 2>&1 | tee "$LOG"
