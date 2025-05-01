# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0




#---- ner ----

python test.py \
--dataset xbrl_finer \
--batch_size 12 \
--quant_bits 8 \
--source hf \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--peft_model ../finetuned_models/finer_train_batched.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8


#--base_model accounts/fireworks/models/llama-v3p1-70b-instruct \
# --peft_model ../src/finetune/OpenFedLLM/output/fingpt-sentiment-train_20000_fedavg_c4s2_i10_b8a1_l512_r8a16_20250116215837/checkpoint-200/
# model gemini-2.0-flash-lite-001
# Sentiment gemini 2.0 lite: projects/1023064188719/locations/us-central1/endpoints/1842980499757203456