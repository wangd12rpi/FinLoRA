# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3



#---- ner ----

python test.py \
--dataset xbrl_finer \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--batch_size 8 \
--quant_bits 8 \
 --peft_model ../finetuned_models/finer_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8 \

