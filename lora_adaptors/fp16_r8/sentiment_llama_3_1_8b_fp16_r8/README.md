
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
micro_batch_size: 8
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
- path: /workspace/FinLoRA/data/train/finlora_sentiment_train.jsonl
  type:
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/sentiment_llama_3_1_8b_fp16_r8
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
wandb_name: sentiment_llama_3_1_8b_fp16_r8

```

</details><br>

# workspace/FinLoRA/lora/axolotl-output/sentiment_llama_3_1_8b_fp16_r8

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/finlora_sentiment_train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2154

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 3
- gradient_accumulation_steps: 2
- total_train_batch_size: 48
- total_eval_batch_size: 24
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0007 | 1    | 3.3646          |
| No log        | 0.2502 | 372  | 0.2299          |
| 0.3256        | 0.5003 | 744  | 0.2189          |
| 0.15          | 0.7505 | 1116 | 0.2159          |
| 0.15          | 1.0007 | 1488 | 0.2168          |
| 0.1247        | 1.2508 | 1860 | 0.2143          |
| 0.0994        | 1.5010 | 2232 | 0.2185          |
| 0.092         | 1.7512 | 2604 | 0.2113          |
| 0.092         | 2.0013 | 2976 | 0.2100          |
| 0.0872        | 2.2515 | 3348 | 0.2138          |
| 0.0722        | 2.5017 | 3720 | 0.2103          |
| 0.0702        | 2.7518 | 4092 | 0.2112          |
| 0.0702        | 3.0020 | 4464 | 0.2092          |
| 0.0695        | 3.2522 | 4836 | 0.2170          |
| 0.0621        | 3.5024 | 5208 | 0.2148          |
| 0.0606        | 3.7525 | 5580 | 0.2154          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1