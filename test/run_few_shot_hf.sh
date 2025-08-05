#!/usr/bin/env bash

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=1
TEMPERATURE=0.0

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_PY="${REPO_ROOT}/test_few_shot.py"
LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/llama8b_hf_few_shot_results_${TS}.txt"
#   fpb fiqa tfns nwgi
#  headline ner
tasks=( financebench xbrl_term formula \
        xbrl_tags_extract xbrl_value_extract xbrl_formula_extract \
        xbrl_formula_calc_extract xbrl_finer xbrl_fnxl 
        cfa_level1 cfa_level2 cfa_level3 cpa_reg)

{
echo "============================================================"
echo "LLAMA‑8B HF Few-Shot $(date)"
echo "============================================================"
for ds in "${tasks[@]}"; do
   echo "DATASET: $ds"
   python "$TEST_PY" --dataset "$ds" --source hf \
          --base_model "$MODEL_NAME" \
          --batch_size "$BATCH_SIZE" --temperature "$TEMPERATURE"
   echo
done
echo "============================================================"
} 2>&1 | tee "$LOG" 