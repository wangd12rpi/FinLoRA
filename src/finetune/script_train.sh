
# Task selection (headline, ner, senti, xbrl)
task="senti"
quant_bits=4
lora_r=4
model_name_short="llama_3.1_8b"  # Can be "llama_3.1_8b" or "llama_3.1_70b"

# Map tasks to dataset paths
declare -A dataset_map=(
  ["headline"]="./data/train/fingpt_headline_train.jsonl"
  ["ner"]="./data/train/fingpt_ner_cls_train.jsonl"
  ["senti"]="../../data/train/finlora_sentiment_train.jsonl"
  ["xbrl_extract"]="../../data/train/xbrl_train.jsonl"
  ["finer"]="../../data/train/finer_train.jsonl"
)



# Map short model names to full Hugging Face model names
declare -A model_map=(
  ["llama_3.1_8b"]="meta-llama/Llama-3.1-8B-Instruct"
  ["llama_3.1_70b"]="meta-llama/Llama-3.1-70B-Instruct"
)


# Start the fine-tuning job in a detached tmux session
tmux new-session -d -s "training_job_${task}" '
export HF_TOKEN=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

  export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_IGNORE_DISABLED_P2P=1
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export TOKENIZERS_PARALLELISM=0
  eval "$(conda shell.bash hook)"

   deepspeed train_lora.py \
    --base_model '"${model_map[$model_name_short]}"' \
    --dataset '"${dataset_map[$task]}"' \
    --max_length 128000 \
    --batch_size 8 \
    --grad_accu 2 \
    --learning_rate 1e-4 \
    --num_epochs 4 \
    --log_interval 10 \
    --warmup_ratio 0.03 \
    --scheduler linear \
    --evaluation_strategy steps \
    --ds_config config_.json \
    --eval_steps 0.05 \
    --quant_bits '"$quant_bits"' \
    --r '"$lora_r"' \
    --max_steps -1 

  read -p "Press Enter to exit..."
'


tmux attach -t "training_job_${task}"
