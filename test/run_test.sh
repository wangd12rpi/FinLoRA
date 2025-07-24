# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0


#---- ner ----

python test.py \
--dataset ner \
--batch_size 1 \
--quant_bits 8 \
--source hf \
--sample_ratio .01 \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--peft_model /workspace/FinLoRA/lora/axolotl-output/general_multi_llama_3_1_8b_8bits_r8

#--base_model accounts/fireworks/models/llama-v3p1-70b-instruct \
# --peft_model ../src/lora/OpenFedLLM/output/fingpt-sentiment-train_20000_fedavg_c4s2_i10_b8a1_l512_r8a16_20250116215837/checkpoint-200/
# model gemini-2.0-flash-lite-001


# Gemini Model names
# Sentiment: projects/1023064188719/locations/us-central1/endpoints/1842980499757203456
# ner projects/1023064188719/locations/us-central1/endpoints/5585682896334618624
# headline projects/1023064188719/locations/us-central1/endpoints/576343104559251456
# finer projects/1023064188719/locations/us-central1/endpoints/7929384289915895808
# xbrl_extract: projects/1023064188719/locations/us-central1/endpoints/8681450243314679808

# financebench projects/1023064188719/locations/us-central1/endpoints/6279905742019362816
# xbrl_term projects/1023064188719/locations/us-central1/endpoints/578348613768314880
# formula projects/1023064188719/locations/us-central1/endpoints/8072338393712820224


#dataset_path = {
#    "xbrl_tags_extract": "../data/test/xbrl_extract_tags_test.jsonl",
#    "xbrl_value_extract": "../data/test/xbrl_extract_value_test.jsonl",
#    "xbrl_formula_extract": "../data/test/xbrl_extract_formula_test.jsonl",
#    "xbrl_formula_calc_extract": "../data/test/xbrl_extract_formula_calculations_test.jsonl",
#    "xbrl_finer": "../data/test/finer_test_batched.jsonl",
#    "xbrl_fnxl": "../data/test/fnxl_test_batched.jsonl",
#    "fpb": "../data/test/fpb_test.jsonl",
#    "fiqa": "../data/test/fiqa_test.jsonl",
#    "tfns": "../data/test/tfns_test.jsonl",
#    "nwgi": "../data/test/nwgi_test.jsonl",
#    "headline": "../data/test/headline_test.jsonl",
#    "ner": "../data/test/ner_test.jsonl",
#    "financebench": "../data/test/financebench_test.jsonl",
#    "xbrl_term": "../data/test/xbrl_term_test.jsonl",
#    "formula": "../data/test/formula_test.jsonl",
#
#}