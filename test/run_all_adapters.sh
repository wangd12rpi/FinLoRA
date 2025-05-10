#!/bin/bash

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=1
TEMPERATURE=0.0
SAMPLE_RATIO=1.0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR="${REPO_ROOT}/test"

mkdir -p "${REPO_ROOT}/test/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${REPO_ROOT}/test/adapter_results_${TIMESTAMP}.txt"

echo "============================================================" > "$RESULTS_FILE"
echo "             ADAPTER EVALUATION RESULTS                     " >> "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"
echo "Base model: $BASE_MODEL" >> "$RESULTS_FILE"
echo "Run time: $(date)" >> "$RESULTS_FILE"
echo "Batch size: $BATCH_SIZE, Temperature: $TEMPERATURE, Sample ratio: $SAMPLE_RATIO" >> "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

test_adapter() {
    local DATASET=$1
    local ADAPTER_CONFIG=$2
    local QUANT_BITS=$3
    
    local ADAPTER_PATH="${REPO_ROOT}/finetuned_models/${DATASET}_llama_3_1_8b_${ADAPTER_CONFIG}"
    local ADAPTER_NAME="${DATASET}_llama_3_1_8b_${ADAPTER_CONFIG}"
    
    echo -e "\n============================================================" | tee -a "$RESULTS_FILE"
    echo "TESTING ADAPTER: $ADAPTER_NAME" | tee -a "$RESULTS_FILE"
    echo "Dataset: $DATASET" | tee -a "$RESULTS_FILE"
    echo "Adapter config: $ADAPTER_CONFIG" | tee -a "$RESULTS_FILE"
    echo "Quantization: $QUANT_BITS bits" | tee -a "$RESULTS_FILE"
    echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"
    
    TEMP_OUTPUT=$(python "${TEST_DIR}/test.py" \
      --dataset $DATASET \
      --batch_size $BATCH_SIZE \
      --source hf \
      --base_model $BASE_MODEL \
      --peft_model $ADAPTER_PATH \
      --temperature $TEMPERATURE \
      --sample_ratio $SAMPLE_RATIO \
      --quant_bits $QUANT_BITS 2>&1)
    
    echo "$TEMP_OUTPUT" | tee -a "$RESULTS_FILE"
    
    ACCURACY=$(echo "$TEMP_OUTPUT" | grep -i "accuracy" | tail -1)
    F1_SCORE=$(echo "$TEMP_OUTPUT" | grep -i "f1" | tail -1)
    
    echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"
    echo "RESULT SUMMARY:" | tee -a "$RESULTS_FILE"
    echo "Adapter: $ADAPTER_NAME" | tee -a "$RESULTS_FILE"
    
    if [ ! -z "$ACCURACY" ]; then
        echo "Accuracy: $ACCURACY" | tee -a "$RESULTS_FILE"
    fi
    
    if [ ! -z "$F1_SCORE" ]; then
        echo "F1 Score: $F1_SCORE" | tee -a "$RESULTS_FILE"
    fi
    
    echo "============================================================" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

echo "Running tests." | tee -a "$RESULTS_FILE"

# Test FinanceBench
echo "TESTING FINANCEBENCH ADAPTERS" | tee -a "$RESULTS_FILE"
test_adapter "financebench" "4bits_r4" 4
test_adapter "financebench" "8bits_r8" 8
test_adapter "financebench" "8bits_r8_dora" 8
test_adapter "financebench" "8bits_r8_rslora" 8

# Test XBRL Term
echo "TESTING XBRL_TERM ADAPTERS" | tee -a "$RESULTS_FILE"
test_adapter "xbrl_term" "4bits_r4" 4
test_adapter "xbrl_term" "8bits_r8" 8
test_adapter "xbrl_term" "8bits_r8_dora" 8
test_adapter "xbrl_term" "8bits_r8_rslora" 8

# Test XBRL formula
echo "TESTING FORMULA ADAPTERS" | tee -a "$RESULTS_FILE"
test_adapter "formula" "4bits_r4" 4
test_adapter "formula" "8bits_r8" 8
test_adapter "formula" "8bits_r8_dora" 8
test_adapter "formula" "8bits_r8_rslora" 8

echo "============================================================" >> "$RESULTS_FILE"
echo "Result Summary" >> "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"

# Accuracy
echo "Accuracy Results:" >> "$RESULTS_FILE"
grep -A 1 "Result Summary:" "$RESULTS_FILE" | grep -i "accuracy" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# F1
echo "F1 Score Results:" >> "$RESULTS_FILE"
grep -A 2 "Result Summary:" "$RESULTS_FILE" | grep -i "f1 score" >> "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"

echo -e "\nAll tests completed. Results saved to $RESULTS_FILE"