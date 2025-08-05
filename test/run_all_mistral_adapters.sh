#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="mistralai/Ministral-8B-Instruct-2410"
BATCH_SIZE=1
TEMPERATURE=0.0
QUANT_BITS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_PY="${REPO_ROOT}/test/test.py"
ADAPTER_DIR="${REPO_ROOT}/lora_adapters/8bits_r8"
LOG_DIR="${SCRIPT_DIR}"
mkdir -p "$LOG_DIR"

# datasets and corresponding Mistral adapters
declare -A dataset_adapter_map=(
    ["financebench"]="financebench_mistral_8b_8bits_r8"
    ["headline"]="headline_mistral_8b_8bits_r8"
    ["ner"]="ner_mistral_8b_8bits_r8"
    ["finer"]="finer_mistral_8b_8bits_r8"
    ["xbrl_tags_extract"]="xbrl_extract_mistral_8b_8bits_r8"
    ["xbrl_value_extract"]="xbrl_extract_mistral_8b_8bits_r8"
    ["xbrl_formula_extract"]="xbrl_extract_mistral_8b_8bits_r8"
    ["xbrl_formula_calc_extract"]="xbrl_extract_mistral_8b_8bits_r8"
    ["xbrl_term"]="xbrl_term_mistral_8b_8bits_r8"
    ["formula"]="formula_mistral_8b_8bits_r8"
    ["fpb"]="sentiment_mistral_8b_8bits_r8"
    ["fiqa"]="sentiment_mistral_8b_8bits_r8"
    ["tfns"]="sentiment_mistral_8b_8bits_r8"
    ["nwgi"]="sentiment_mistral_8b_8bits_r8"
    ["fnxl"]="finer_mistral_8b_8bits_r8"
    ["cpa_reg"]="regulations_mistral_8b_8bits_r8"
    ["cfa_level1"]="regulations_mistral_8b_8bits_r8"
    ["cfa_level2"]="regulations_mistral_8b_8bits_r8"
    ["cfa_level3"]="regulations_mistral_8b_8bits_r8"
)

TS="$(date +%Y%m%d_%H%M%S)_mistral"
RESULTS_FILE="${LOG_DIR}/mistral_adapter_results_${TS}.txt"

{
    echo "============================================================"
    echo "           MISTRAL ADAPTER EVALUATION RUN"
    echo "Base model  : $BASE_MODEL"
    echo "Run started : $(date)"
    echo "Batch size  : $BATCH_SIZE   Temp: $TEMPERATURE"
    echo "Quant bits  : $QUANT_BITS"
    echo "============================================================"
    echo
} | tee "$RESULTS_FILE"

# Test all datasets
echo "Testing all Mistral adapters" | tee -a "$RESULTS_FILE"

for dataset in "${!dataset_adapter_map[@]}"; do
    adapter_name="${dataset_adapter_map[$dataset]}"
    peft_path="${ADAPTER_DIR}/${adapter_name}"
    
    {
        echo
        echo "------------------------------------------------------------"
        echo "Dataset     : $dataset"
        echo "Adapter     : $adapter_name"
        echo "PEFT model  : $peft_path"
        echo "------------------------------------------------------------"
    } | tee -a "$RESULTS_FILE"
    
    (
        cd "$REPO_ROOT"
        python "$TEST_PY" \
            --dataset "$dataset" \
            --batch_size "$BATCH_SIZE" \
            --quant_bits "$QUANT_BITS" \
            --source hf \
            --sample_ratio 1.0 \
            --base_model "$BASE_MODEL" \
            --peft_model "$peft_path" \
            --temperature "$TEMPERATURE" \
        2>&1
    ) | tee -a "$RESULTS_FILE"
    
    echo | tee -a "$RESULTS_FILE"
done

{
    echo "Run finished : $(date)"
    echo "Log saved to : $RESULTS_FILE"
    echo
    echo "Summary:"
    echo "- ${#dataset_adapter_map[@]} total datasets tested"
    echo "- Base model: $BASE_MODEL"
    echo "- Configuration: 8bits_r8"
} | tee -a "$RESULTS_FILE" 