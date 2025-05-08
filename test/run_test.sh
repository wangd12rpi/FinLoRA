# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0




#---- ner ----

python test.py \
--dataset headline \
--batch_size 1 \
--quant_bits 8 \
--source google \
--sample_ratio .1 \
--base_model projects/1023064188719/locations/us-central1/endpoints/576343104559251456 \
#--base_model meta-llama/Llama-3.1-8B \
#--peft_model ../finetuned_models/finer_llama_3_1_8B_8bits_r8

#--base_model accounts/fireworks/models/llama-v3p1-70b-instruct \
# --peft_model ../src/lora/OpenFedLLM/output/fingpt-sentiment-train_20000_fedavg_c4s2_i10_b8a1_l512_r8a16_20250116215837/checkpoint-200/
# model gemini-2.0-flash-lite-001


# Gemini Model names
# Sentiment: projects/1023064188719/locations/us-central1/endpoints/1842980499757203456
# ner projects/1023064188719/locations/us-central1/endpoints/5585682896334618624
# headline projects/1023064188719/locations/us-central1/endpoints/576343104559251456
# finer projects/1023064188719/locations/us-central1/endpoints/7929384289915895808
