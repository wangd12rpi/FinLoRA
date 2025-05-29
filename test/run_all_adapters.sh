#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=1
TEMPERATURE=0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_PY="${REPO_ROOT}/test/test.py"
MODEL_DIR="${REPO_ROOT}/lora_models"
LOG_DIR="${SCRIPT_DIR}"
mkdir -p "$LOG_DIR"

datasets=(
  xbrl_tags_extract xbrl_value_extract xbrl_formula_extract xbrl_formula_calc_extract
  xbrl_finer xbrl_fnxl
  fpb fiqa tfns nwgi
  headline ner financebench xbrl_term formula
)

configs=(4bits_r4 8bits_r8 8bits_r8_dora 8bits_r8_rslora)
quant_bits() { [[ $1 == 4bits* ]] && echo 4 || echo 8; }

adapter_prefix() {
  case "$1" in
    financebench) echo financebench ;;
    xbrl_term) echo xbrl_term ;;
    formula) echo formula ;;
    xbrl_tags_extract|xbrl_value_extract|xbrl_formula_extract|xbrl_formula_calc_extract) echo xbrl_extract ;;
    xbrl_finer|xbrl_fnxl) echo finer ;;
    fpb|fiqa|tfns|nwgi) echo sentiment ;;
    headline) echo headline ;;
    ner) echo ner ;;
    *) echo "$1" ;;
  esac
}

TS="$(date +%Y%m%d_%H%M%S)_run1"
RESULTS_FILE="${LOG_DIR}/adapter_results_${TS}.txt"

{
  echo "============================================================"
  echo "               ADAPTER EVALUATION RUN #1"
  echo "Base model  : $BASE_MODEL"
  echo "Run started : $(date)"
  echo "Batch size  : $BATCH_SIZE   Temp: $TEMPERATURE"
  echo "Sampling    : Full (sample_ratio=1.0)"
  echo "============================================================"
  echo
} | tee "$RESULTS_FILE"

for ds in "${datasets[@]}"; do
  SAMPLE_RATIO=1.0

  echo "##############  DATASET: $ds  (sample_ratio=$SAMPLE_RATIO)  ##############" | tee -a "$RESULTS_FILE"

  for cfg in "${configs[@]}"; do
    prefix=$(adapter_prefix "$ds")
    peft_path="${MODEL_DIR}/${cfg}/${prefix}_llama_3_1_8b_${cfg}"
    qb=$(quant_bits "$cfg")

    {
      echo
      echo "------------------------------------------------------------"
      echo "Adapter cfg : $cfg   (quant ${qb}-bit)"
      echo "PEFT model  : $peft_path"
      echo "------------------------------------------------------------"
    } | tee -a "$RESULTS_FILE"

    (
      cd "$REPO_ROOT"
      python "$TEST_PY" \
        --dataset "$ds" \
        --batch_size "$BATCH_SIZE" \
        --quant_bits "$qb" \
        --source hf \
        --sample_ratio "$SAMPLE_RATIO" \
        --base_model "$BASE_MODEL" \
        --peft_model "$peft_path" \
        --temperature "$TEMPERATURE" \
      2>&1
    ) | tee -a "$RESULTS_FILE"
  done
  echo | tee -a "$RESULTS_FILE"
done

{
  echo "============================================================"
  echo "Run finished : $(date)"
  echo "Log saved to : $RESULTS_FILE"
  echo
} | tee -a "$RESULTS_FILE"
