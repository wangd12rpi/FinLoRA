---
library_name: peft
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- generated_from_trainer
datasets:
- /workspace/FinLoRA/data/train/finer_train_batched.jsonl
model-index:
- name: workspace/FinLoRA/lora/axolotl-output/finer_llama_3_1_8b_fp16_r8
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.9.1.post1`
```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 4
learning_rate: 0.0001
optimizer: adamw_torch_fused
lr_scheduler: cosine
load_in_8bit: false
load_in_4bit: false
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
datasets:
- path: /workspace/FinLoRA/data/train/finer_train_batched.jsonl
  type:
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/finer_llama_3_1_8b_fp16_r8
sequence_len: 4096
gradient_checkpointing: true
logging_steps: 500
warmup_steps: 10
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.0
special_tokens:
  pad_token: <|end_of_text|>
deepspeed: deepspeed_configs/zero1.json
bf16: auto
tf32: false
chat_template: llama3
wandb_name: finer_llama_3_1_8b_fp16_r8

```

</details><br>

# workspace/FinLoRA/lora/axolotl-output/finer_llama_3_1_8b_fp16_r8

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/finer_train_batched.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0345

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
- num_devices: 3
- gradient_accumulation_steps: 8
- total_train_batch_size: 24
- total_eval_batch_size: 3
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0024 | 1    | 0.5279          |
| No log        | 0.2498 | 102  | 0.0542          |
| No log        | 0.4995 | 204  | 0.0469          |
| No log        | 0.7493 | 306  | 0.0425          |
| No log        | 0.9991 | 408  | 0.0391          |
| 0.0635        | 1.2498 | 510  | 0.0410          |
| 0.0635        | 1.4995 | 612  | 0.0396          |
| 0.0635        | 1.7493 | 714  | 0.0372          |
| 0.0635        | 1.9991 | 816  | 0.0373          |
| 0.0635        | 2.2498 | 918  | 0.0355          |
| 0.0326        | 2.4995 | 1020 | 0.0339          |
| 0.0326        | 2.7493 | 1122 | 0.0355          |
| 0.0326        | 2.9991 | 1224 | 0.0335          |
| 0.0326        | 3.2498 | 1326 | 0.0345          |
| 0.0326        | 3.4995 | 1428 | 0.0347          |
| 0.0247        | 3.7493 | 1530 | 0.0344          |
| 0.0247        | 3.9991 | 1632 | 0.0345          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1