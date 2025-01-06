# FinLoRA Test Scripts

### Test a specific dataset

Run the following python script or use `run_test.sh`.
```bash
python test.py \
--dataset xbrl_finer \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--batch_size 8 \
--quant_bits 8 \
--peft_model ../finetuned_models/finer_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8 \
```

### Test all existing LoRA adaptors

Run the following python script to test all LoRA adaptors in `../finetuned_models`, or use `run_test_all.sh`.
```bash
python python test_all.py
```

