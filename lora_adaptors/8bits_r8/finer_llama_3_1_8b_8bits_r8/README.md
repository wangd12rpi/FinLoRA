
<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.9.0`
```yaml
base_model: NousResearch/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0001
load_in_8bit: true
load_in_4bit: false
adapter: lora
lora_model_dir: null
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
- q_proj
- v_proj
- k_proj
datasets:
- path: /workspace/FinLoRA/data/train/finer_train_batched.jsonl
  type:
    system_prompt: ''
    field_system: system
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
dataset_prepared_path: null
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/finer_llama_3_1_8b_8bits_r8
sequence_len: 4096
sample_packing: false
pad_to_sequence_len: false
wandb_project: finlora_models
wandb_entity: null
wandb_watch: gradients
wandb_name: finer_llama_3_1_8b_8bits_r8
wandb_log_model: 'false'
bf16: auto
tf32: false
gradient_checkpointing: true
resume_from_checkpoint: null
logging_steps: 500
flash_attention: false
deepspeed: deepspeed_configs/zero1.json
warmup_steps: 10
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.0
special_tokens:
  pad_token: <|end_of_text|>
chat_template: llama3

```

</details><br>

# workspace/FinLoRA/fine-tune/axolotl-output/finer_llama_3_1_8B_8bits_r8

This model is a fine-tuned version of [NousResearch/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/finer_train_batched.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0331

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- total_eval_batch_size: 2
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0016 | 1    | 0.5433          |
| No log        | 0.2497 | 153  | 0.0520          |
| No log        | 0.4995 | 306  | 0.0459          |
| No log        | 0.7492 | 459  | 0.0406          |
| 0.0693        | 0.9990 | 612  | 0.0386          |
| 0.0693        | 1.2497 | 765  | 0.0396          |
| 0.0693        | 1.4995 | 918  | 0.0363          |
| 0.036         | 1.7492 | 1071 | 0.0351          |
| 0.036         | 1.9990 | 1224 | 0.0348          |
| 0.036         | 2.2497 | 1377 | 0.0360          |
| 0.0302        | 2.4995 | 1530 | 0.0321          |
| 0.0302        | 2.7492 | 1683 | 0.0347          |
| 0.0302        | 2.9990 | 1836 | 0.0324          |
| 0.0302        | 3.2497 | 1989 | 0.0328          |
| 0.0242        | 3.4995 | 2142 | 0.0334          |
| 0.0242        | 3.7492 | 2295 | 0.0332          |
| 0.0242        | 3.9990 | 2448 | 0.0331          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.0
- Tokenizers 0.21.1