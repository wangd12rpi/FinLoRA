# FinLoRA Test Scripts

### Test a specific dataset

Run the following python script or use `run_test.sh`.

```bash
python test.py \
--dataset xbrl_finer \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--batch_size 8 \
--quant_bits 8 \
--peft_model ../lora_adapters/8b_8bits_r8/finer_non_batch_llama_3_1_8b_8bits_r8 \
```

### Test all existing LoRA adapters

Run the following python script to test all LoRA adapters in `../lora_models`, or use `run_test_all.sh`.

```bash
python python test_all.py
```
