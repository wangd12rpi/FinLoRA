#!/usr/bin/env bash
TOGETHER_API_KEY="${TOGETHER_API_KEY:-api_key}"
export TOGETHER_API_KEY

MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
BATCH_SIZE=1
TEMPERATURE=0.0

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_PY="${REPO_ROOT}/test.py"
LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/llama70b_together_results_${TS}.txt"

tasks=( financebench xbrl_term formula \
        xbrl_tags_extract xbrl_value_extract xbrl_formula_extract \
        xbrl_formula_calc_extract xbrl_finer xbrl_fnxl )

{
echo "============================================================"
echo "LLAMA‑70B Together $(date)"
echo "============================================================"
for ds in "${tasks[@]}"; do
   echo "DATASET: $ds"
   python "$TEST_PY" --dataset "$ds" --source together \
          --model_name "$MODEL_NAME" --together_api_key "$TOGETHER_API_KEY" \
          --batch_size "$BATCH_SIZE" --temperature "$TEMPERATURE"
   echo
done
echo "============================================================"
} 2>&1 | tee "$LOG"
