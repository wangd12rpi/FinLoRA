
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
- path: /workspace/FinLoRA/data/train/ner_train.jsonl
  type:
    system_prompt: ''
    field_system: system
    field_instruction: context
    field_output: target
    format: '[INST] {instruction} [/INST]'
    no_input_format: '[INST] {instruction} [/INST]'
dataset_prepared_path: null
val_set_size: 0.02
output_dir: /workspace/FinLoRA/lora/axolotl-output/ner_llama_3_1_8b_8bits_r8_rslora
peft_use_dora: false
peft_use_rslora: true
sequence_len: 4096
sample_packing: false
pad_to_sequence_len: false
wandb_project: finlora_models
wandb_entity: null
wandb_watch: gradients
wandb_name: ner_llama_3_1_8b_8bits_r8_rslora
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

# workspace/FinLoRA/lora/axolotl-output/ner_llama_3_1_8b_8bits_r8_rslora

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the /workspace/FinLoRA/data/train/ner_train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1134

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
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0036 | 1    | 8.5107          |
| No log        | 0.2527 | 70   | 0.1374          |
| No log        | 0.5054 | 140  | 0.0055          |
| No log        | 0.7581 | 210  | 0.1240          |
| No log        | 1.0108 | 280  | 0.1526          |
| No log        | 1.2635 | 350  | 0.1392          |
| No log        | 1.5162 | 420  | 0.1319          |
| No log        | 1.7690 | 490  | 0.1209          |
| 0.1301        | 2.0217 | 560  | 0.1246          |
| 0.1301        | 2.2744 | 630  | 0.1351          |
| 0.1301        | 2.5271 | 700  | 0.1100          |
| 0.1301        | 2.7798 | 770  | 0.1228          |
| 0.1301        | 3.0325 | 840  | 0.1465          |
| 0.1301        | 3.2852 | 910  | 0.1353          |
| 0.1301        | 3.5379 | 980  | 0.0954          |
| 0.0           | 3.7906 | 1050 | 0.1134          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0.dev20250319+cu128
- Datasets 3.5.1
- Tokenizers 0.21.1