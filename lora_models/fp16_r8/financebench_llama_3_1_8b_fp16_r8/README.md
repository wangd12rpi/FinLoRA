
<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.9.1.post1`
```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
gradient_accumulation_steps: 2
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
- path: /workspace/FinLoRA/data/train/financebench_train.jsonl
  type:
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_fp16_r8
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
wandb_name: financebench_llama_3_1_8b_fp16_r8

```

</details><br>

# workspace/FinLoRA/lora/axolotl-output/financebench_llama_3_1_8b_fp16_r8

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/financebench_train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 2.2962

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
- gradient_accumulation_steps: 2
- total_train_batch_size: 6
- total_eval_batch_size: 3
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0714 | 1    | 4.7562          |
| No log        | 0.2857 | 4    | 4.7251          |
| No log        | 0.5714 | 8    | 4.4087          |
| No log        | 0.8571 | 12   | 3.9440          |
| No log        | 1.1429 | 16   | 3.3303          |
| No log        | 1.4286 | 20   | 2.8647          |
| No log        | 1.7143 | 24   | 2.6231          |
| No log        | 2.0    | 28   | 2.4685          |
| No log        | 2.2857 | 32   | 2.4041          |
| No log        | 2.5714 | 36   | 2.3627          |
| No log        | 2.8571 | 40   | 2.3343          |
| No log        | 3.1429 | 44   | 2.3287          |
| No log        | 3.4286 | 48   | 2.3136          |
| No log        | 3.7143 | 52   | 2.3032          |
| No log        | 4.0    | 56   | 2.2962          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1