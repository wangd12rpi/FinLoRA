#!/usr/bin/env bash
# Evaluate financial tasks with the alpaca‑mixlora‑7b adapter (Llama‑2‑7B).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" 
DATA_DIR="$PROJECT_ROOT/data/test"

MIXLORA_MODEL_PATH="TUDB-Labs/alpaca-mixlora-7b"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAMPLE_RATIO=1.0
BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MIXLORA_MODEL_PATH="$2"; shift 2 ;;
    --base-model)   BASE_MODEL="$2";         shift 2 ;;
    --sample-ratio) SAMPLE_RATIO="$2";       shift 2 ;;
    --batch-size)   BATCH_SIZE="$2";         shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "$MIXLORA_MODEL_PATH" ]] && { echo "Error: --model is required"; exit 1; }

mkdir -p "$SCRIPT_DIR/results"
RUN_LOG="$SCRIPT_DIR/financial_mixlora_results_$(date +%Y%m%d_%H%M%S).txt"

echo -e "\n===== MixLoRA financial benchmark =====" | tee -a "$RUN_LOG"
echo "Adapter   : $MIXLORA_MODEL_PATH"            | tee -a "$RUN_LOG"
echo "Base model: $BASE_MODEL"                    | tee -a "$RUN_LOG"
echo "Batch     : $BATCH_SIZE   Sample: $SAMPLE_RATIO" | tee -a "$RUN_LOG"
echo "=========================================" | tee -a "$RUN_LOG"

# Set environment variable to disable TF warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Task 3: NER
echo -e "\nTASK 3 — NER" | tee -a "$RUN_LOG"
python "$SCRIPT_DIR/test_mixlora_models.py" \
    --model_path "$MIXLORA_MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --sample_ratio "$SAMPLE_RATIO" \
    --batch_size "$BATCH_SIZE" \
    --task_type "ner" \
    --dataset "$DATA_DIR/ner_test.jsonl" 2>&1 | tee -a "$RUN_LOG"

# Task 4: FINER 
echo -e "\nTASK 4 — FINER" | tee -a "$RUN_LOG"
python "$SCRIPT_DIR/test_mixlora_models.py" \
    --model_path "$MIXLORA_MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --sample_ratio "$SAMPLE_RATIO" \
    --batch_size "$BATCH_SIZE" \
    --task_type "finer" \
    --dataset "$DATA_DIR/finer_test_batched.jsonl" 2>&1 | tee -a "$RUN_LOG"

# Task 5: FNXL
echo -e "\nTASK 5 — FNXL" | tee -a "$RUN_LOG"
python "$SCRIPT_DIR/test_mixlora_models.py" \
    --model_path "$MIXLORA_MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --sample_ratio "$SAMPLE_RATIO" \
    --batch_size "$BATCH_SIZE" \
    --task_type "fnxl" \
    --dataset "$DATA_DIR/fnxl_test_batched.jsonl" 2>&1 | tee -a "$RUN_LOG"

echo -e "\nAll tasks finished. Log saved to $RUN_LOG" 