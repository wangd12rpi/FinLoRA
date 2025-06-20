
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
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 0.0001
load_in_8bit: false
load_in_4bit: true
adapter: lora
lora_model_dir: null
lora_r: 4
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
output_dir: /workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_4bits_r4
peft_use_dora: false
sequence_len: 4096
sample_packing: false
pad_to_sequence_len: false
wandb_project: finlora_models
wandb_entity: null
wandb_watch: gradients
wandb_name: financebench_llama_3_1_8b_4bits_r4
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

# workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_4bits_r4

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/financebench_train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 3.3003

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
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.1176 | 1    | 4.9794          |
| No log        | 0.2353 | 2    | 4.9922          |
| No log        | 0.4706 | 4    | 4.9603          |
| No log        | 0.7059 | 6    | 4.8793          |
| No log        | 0.9412 | 8    | 4.6411          |
| No log        | 1.1176 | 10   | 4.4789          |
| No log        | 1.3529 | 12   | 4.1465          |
| No log        | 1.5882 | 14   | 3.9720          |
| No log        | 1.8235 | 16   | 3.8714          |
| No log        | 2.0    | 18   | 3.7423          |
| No log        | 2.2353 | 20   | 3.6258          |
| No log        | 2.4706 | 22   | 3.5165          |
| No log        | 2.7059 | 24   | 3.4236          |
| No log        | 2.9412 | 26   | 3.3368          |
| No log        | 3.1176 | 28   | 3.3172          |
| No log        | 3.3529 | 30   | 3.2741          |
| No log        | 3.5882 | 32   | 3.3003          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1