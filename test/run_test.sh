# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3



#---- ner ----

python test.py \
--dataset xbrl_finer \
--batch_size 8 \
--quant_bits 8 \
#--base_model accounts/fireworks/models/llama-v3p1-70b-instruct \
# --peft_model ../src/finetune/OpenFedLLM/output/fingpt-sentiment-train_20000_fedavg_c4s2_i10_b8a1_l512_r8a16_20250116215837/checkpoint-200/
