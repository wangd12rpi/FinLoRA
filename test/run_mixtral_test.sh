#!/bin/bash

# Test Mixtral model on all non-XBRL tasks using Together API API

TOGETHER_API_KEY="api key here"
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
BATCH_SIZE=8
TEMPERATURE=0.0
SAMPLE_RATIO=1.0

echo "Testing $MODEL on all non-XBRL tasks using Together API only (no local model loading)"

mkdir -p test/results

cd test

# # Financial sentiment analysis tasks
# echo "Testing FPB (Financial PhraseBank)"
# python test.py \
#   --dataset fpb \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# echo "Testing FIQA"
# python test.py \
#   --dataset fiqa \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# echo "Testing TFNS"
# python test.py \
#   --dataset tfns \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# # Other finance tasks
# echo "Testing NWGI"
# python test.py \
#   --dataset nwgi \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# echo "Testing Headline"
# python test.py \
#   --dataset headline \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# echo "Testing NER"
# python test.py \
#   --dataset ner \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

# echo "Testing FNXL"
# python test.py \
#   --dataset xbrl_fnxl \
#   --batch_size $BATCH_SIZE \
#   --source together \
#   --together_api_key $TOGETHER_API_KEY \
#   --base_model $MODEL \
#   --temperature $TEMPERATURE \
#   --sample_ratio $SAMPLE_RATIO \
#   --peft_model ""

echo "Testing FiNER"
python test.py \
  --dataset xbrl_finer \
  --batch_size $BATCH_SIZE \
  --source together \
  --together_api_key $TOGETHER_API_KEY \
  --base_model $MODEL \
  --temperature $TEMPERATURE \
  --sample_ratio $SAMPLE_RATIO \
  --peft_model ""

echo "All non-XBRL tasks completed for $MODEL" 