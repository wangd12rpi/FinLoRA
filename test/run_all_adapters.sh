#!/usr/bin/env bash
# run_adapters.sh – loop over adapters and save full output to a log file

###############################################################################
# Change these if needed
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=1
TEMPERATURE=0.0
SAMPLE_RATIO=1.0
###############################################################################

# --- Repo layout -------------------------------------------------------------
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_PY="${REPO_ROOT}/test.py"          # adjust if test.py lives elsewhere
MODEL_DIR="${REPO_ROOT}/finetuned_models"   # parent of all adapter folders
LOG_DIR="${REPO_ROOT}/test"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${LOG_DIR}/adapter_results_${TIMESTAMP}.txt"
# -----------------------------------------------------------------------------

# Datasets and adapter configs you want to run
datasets=( financebench xbrl_term formula )
configs=( 4bits_r4 8bits_r8 8bits_r8_dora 8bits_r8_rslora )

# Helper: derive quant bits from config name
quant_bits() { [[ $1 == 4bits* ]] && echo 4 || echo 8; }

# ----------------------------------------------------------------------------- #
# Start logging
echo "============================================================"      | tee  "$RESULTS_FILE"
echo "               ADAPTER EVALUATION RUN                       "  | tee -a "$RESULTS_FILE"
echo "Base model  : $BASE_MODEL"                                   | tee -a "$RESULTS_FILE"
echo "Run started : $(date)"                                       | tee -a "$RESULTS_FILE"
echo "Batch size  : $BATCH_SIZE   Temp: $TEMPERATURE   Sample: $SAMPLE_RATIO" \
                                                                 | tee -a "$RESULTS_FILE"
echo "============================================================"      | tee -a "$RESULTS_FILE"
echo ""                                                              | tee -a "$RESULTS_FILE"

for ds in "${datasets[@]}"; do
  echo "################  DATASET: ${ds}  ################"             | tee -a "$RESULTS_FILE"
  for cfg in "${configs[@]}"; do
    peft_path="${MODEL_DIR}/${ds}_llama_3_1_8b_${cfg}"
    qb=$(quant_bits "$cfg")

    echo -e "\n============================================================" | tee -a "$RESULTS_FILE"
    echo "Adapter cfg : ${cfg}   (quant ${qb}‑bit)"                     | tee -a "$RESULTS_FILE"
    echo "PEFT model  : ${peft_path}"                                   | tee -a "$RESULTS_FILE"
    echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"

    # Run test.py and capture both stdout & stderr, appending to log
    python "$TEST_PY" \
      --dataset "$ds" \
      --batch_size "$BATCH_SIZE" \
      --quant_bits "$qb" \
      --source hf \
      --sample_ratio "$SAMPLE_RATIO" \
      --base_model "$BASE_MODEL" \
      --peft_model "$peft_path" \
      --temperature "$TEMPERATURE" 2>&1 | tee -a "$RESULTS_FILE"

  done
  echo                                                             | tee -a "$RESULTS_FILE"
done

echo "============================================================"      | tee -a "$RESULTS_FILE"
echo "Run finished : $(date)"                                         | tee -a "$RESULTS_FILE"
echo "Log saved to  : $RESULTS_FILE"
