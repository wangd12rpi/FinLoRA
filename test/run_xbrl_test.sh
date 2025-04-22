#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create .env file if it doesn't exist
if [ ! -f "${ROOT_DIR}/.env" ]; then
    echo "Creating .env file with Together AI API key"
    echo "TOGETHER_API_KEY=TOKEN" > "${ROOT_DIR}/.env"
fi

SAMPLE_RATIO=1.0
API_KEY="TOKEN"
MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
TEMPERATURE=0.0

echo "Running XBRL Extraction tasks using tqdm for progress reporting"
echo "Using full dataset with temperature=${TEMPERATURE}"
echo "Model: ${MODEL}"

# echo "Running xbrl_finer..."
# python test_dataset.py --dataset "xbrl_finer" --base_model "${MODEL}" --together_api_key "${API_KEY}" --sample_ratio ${SAMPLE_RATIO} --temperature ${TEMPERATURE}

# echo "Running xbrl_fnxl..."
# python test_dataset.py --dataset "xbrl_fnxl" --base_model "${MODEL}" --together_api_key "${API_KEY}" --sample_ratio ${SAMPLE_RATIO} --temperature ${TEMPERATURE}

echo "Running xbrl_tags_extract..."
python xbrl.py --dataset "xbrl_tags_extract" --base_model "${MODEL}" --together_api_key "${API_KEY}" --sample_ratio ${SAMPLE_RATIO} --temperature ${TEMPERATURE}

# echo "Running xbrl_value_extract..."
# python test_dataset.py --dataset "xbrl_value_extract" --base_model "${MODEL}" --together_api_key "${API_KEY}" --sample_ratio ${SAMPLE_RATIO} --temperature ${TEMPERATURE}

# echo "Running xbrl_formula_extract..."
# python test_dataset.py --dataset "xbrl_formula_extract" --base_model "${MODEL}" --together_api_key "${API_KEY}" --sample_ratio ${SAMPLE_RATIO} --temperature ${TEMPERATURE}

echo "All XBRL extraction tasks completed"
