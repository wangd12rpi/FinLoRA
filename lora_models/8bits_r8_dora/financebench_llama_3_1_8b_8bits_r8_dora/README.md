
<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.9.1`
```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
gradient_accumulation_steps: 2
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
- path: /workspace/FinLoRA/data/train/financebench_train.jsonl
  type:
    system_prompt: ''
    field_system: system
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
dataset_prepared_path: null
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_8bits_r8_dora
peft_use_dora: true
peft_use_rslora: false
sequence_len: 4096
sample_packing: false
pad_to_sequence_len: false
wandb_project: finlora_models
wandb_entity: null
wandb_watch: gradients
wandb_name: financebench_llama_3_1_8b_8bits_r8_dora
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

# workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_8bits_r8_dora

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/financebench_train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 3.0309

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
- num_devices: 5
- gradient_accumulation_steps: 2
- total_train_batch_size: 10
- total_eval_batch_size: 5
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.1176 | 1    | 4.7299          |
| No log        | 0.2353 | 2    | 4.7794          |
| No log        | 0.4706 | 4    | 4.7367          |
| No log        | 0.7059 | 6    | 4.6608          |
| No log        | 0.9412 | 8    | 4.5025          |
| No log        | 1.1176 | 10   | 4.2269          |
| No log        | 1.3529 | 12   | 3.9951          |
| No log        | 1.5882 | 14   | 3.7253          |
| No log        | 1.8235 | 16   | 3.5911          |
| No log        | 2.0    | 18   | 3.4499          |
| No log        | 2.2353 | 20   | 3.4191          |
| No log        | 2.4706 | 22   | 3.2604          |
| No log        | 2.7059 | 24   | 3.1056          |
| No log        | 2.9412 | 26   | 2.9699          |
| No log        | 3.1176 | 28   | 3.0718          |
| No log        | 3.3529 | 30   | 3.0529          |
| No log        | 3.5882 | 32   | 3.0309          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1