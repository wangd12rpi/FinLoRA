import json
import yaml
import subprocess
import argparse
import os
import sys
from pathlib import Path

# --- YAML Template ---
# Define the base Axolotl YAML configuration structure as a Python dictionary
# We will override parts of this with values from the JSON config
AXOLOTL_YAML_TEMPLATE = {
    "base_model": "placeholder-model",  # Placeholder, will be overridden
    "model_type": "AutoModelForCausalLM",  # Will be set based on model type
    "tokenizer_type": "AutoTokenizer",
    "gradient_accumulation_steps": 4,  # Placeholder
    "micro_batch_size": 2,  # Placeholder
    "num_epochs": 4,  # Placeholder
    "optimizer": "adamw_bnb_8bit",  # Good default for quantized models
    "lr_scheduler": "cosine",
    "learning_rate": 0.0002,  # Placeholder

    "load_in_8bit": False,  # Placeholder
    "load_in_4bit": False,  # Placeholder
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "adapter": "lora",
    "lora_model_dir": None,  # Usually not needed unless merging later
    "lora_r": 8,  # Placeholder
    "lora_alpha": 16,  # Default, can be overridden if needed in JSON
    "lora_dropout": 0.05,  # Default
    "lora_target_modules": [  # Default for Llama, adjust if needed for other models
        "q_proj",
        "v_proj",
        "k_proj",
        # "o_proj", # Often included
        # "gate_proj", # Often included
        # "up_proj", # Often included
        # "down_proj" # Often included
    ],

    # "chat_template": "llama3",  # Default, adjust if model needs different template
    "datasets": [
        {
            "path": "../data/train/placeholder_dataset.jsonl",  # Placeholder
            "type": {
                # --- Instruction Fine-Tuning Format ---
                # Adjust field_system, field_instruction, field_output, and format
                # based on your dataset's structure and how you want prompts formatted.
                "system_prompt": "",  # Optional system prompt
                "field_system": "system",  # Field name in JSONL for system message (if any)
                "field_instruction": "context",  # Field name for the instruction/input/context
                "field_output": "target",  # Field name for the desired response/target
                "format": "[INST] {instruction} [/INST]",  # Format for instruction-only
                "no_input_format": "[INST] {instruction} [/INST]"  # Same if no separate input field
            }
        }
    ],

    "dataset_prepared_path": None,  # Let Axolotl handle preparation
    "val_set_size": 0.02,  # Default validation split
    "output_dir": "./axolotl-output/placeholder-run",  # Placeholder
    "peft_use_dora": False,
    "peft_use_rslora": False,

    "sequence_len": 4096,  # Max length of input sequence
    "sample_packing": False,
    "pad_to_sequence_len": False,

    # --- WandB Config (Optional) ---
    "wandb_project": "finlora_models",  # Your project name
    "wandb_entity": None,  # Your WandB username or team (optional)
    "wandb_watch": "gradients",  # options: false, gradients, parameters, all (watch consumes memory)
    "wandb_name": None,
    "wandb_log_model": "false",  # options: false, true, checkpoint, end

    # --- Performance & Precision ---
    "bf16": "auto",  # Use BF16 if available, otherwise FP16
    "tf32": False,  # Usually False for Ampere+ GPUs

    # --- Training Optimizations ---
    "gradient_checkpointing": True,
    "resume_from_checkpoint": None,  # Set to path or True to resume
    "logging_steps": 500,
    "flash_attention": False,  # Use Flash Attention if available

    # --- DeepSpeed (Optional) ---
    # Ensure the deepspeed config file exists if specified
    "deepspeed": "deepspeed_configs/zero1.json",  # Example, adjust path/config

    # --- Scheduler & Saving ---
    "warmup_steps": 10,
    "evals_per_epoch": 4,  # How often to evaluate on validation set
    "saves_per_epoch": 1,  # How often to save checkpoints
    "weight_decay": 0.0,
    "special_tokens": {  # Important for Llama 3 and others
        "pad_token": "<|end_of_text|>"
    }
}


def load_config(json_path):
    """Loads the JSON configuration file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON configuration file not found at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        sys.exit(1)


def generate_axolotl_yaml(run_config, run_name, template):
    """Generates the specific Axolotl YAML config dictionary for a run."""
    yaml_config = template.copy()  # Start with the template
    # --- Override template values with JSON config ---
    yaml_config['base_model'] = run_config.get('base_model', template['base_model'])
    yaml_config['gradient_accumulation_steps'] = run_config.get('gradient_accumulation_steps',
                                                                template['gradient_accumulation_steps'])
    yaml_config['micro_batch_size'] = run_config.get('batch_size', template[
        'micro_batch_size'])  # Map json 'batch_size' to yaml 'micro_batch_size'
    yaml_config['num_epochs'] = run_config.get('num_epochs', template['num_epochs'])
    yaml_config['learning_rate'] = run_config.get('learning_rate', template['learning_rate'])
    yaml_config['lora_r'] = run_config.get('lora_r', template['lora_r'])

    yaml_config['peft_use_dora'] = run_config.get('peft_use_dora', template['peft_use_dora'])
    yaml_config['peft_use_rslora'] = run_config.get('peft_use_rslora', template['peft_use_rslora'])
    print(f"\n\n*****USING DORA?********: {yaml_config['peft_use_dora']}", "\n\n******************")
    print(f"\n\n*****USING rsLoRA?********: {yaml_config['peft_use_rslora']}", "\n\n******************")

    # Handle Quantization
    quant_bits = run_config.get('quant_bits')
    if quant_bits == 8:
        yaml_config['load_in_8bit'] = True
        yaml_config['load_in_4bit'] = False
        yaml_config['optimizer'] = 'adamw_bnb_8bit'
        yaml_config['bnb_4bit_use_double_quant'] = False
        yaml_config['bnb_4bit_quant_type'] = None
        yaml_config['bnb_4bit_compute_dtype'] = None

    elif quant_bits == 4:
        yaml_config['load_in_8bit'] = False
        yaml_config['load_in_4bit'] = True
        yaml_config['bnb_4bit_use_double_quant'] = True
        yaml_config['bnb_4bit_quant_type'] = "nf4"
        yaml_config['bnb_4bit_compute_dtype'] = "bfloat16"
        yaml_config['gradient_checkpointing'] = True
        yaml_config['optimizer'] = 'paged_adamw_8bit'
    else:
        print("Error: 'quant_bits' must be either 8 or 4.'")
        sys.exit(1)

    # Dataset path
    if 'dataset_path' in run_config:
        dataset_path = Path(run_config['dataset_path']).resolve()
        yaml_config['datasets'][0]['path'] = str(dataset_path)
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} does not seem to exist.")
    else:
        print("Error: 'dataset_path' not specified in the config for this run.")
        sys.exit(1)

    # Output directory - use the run name
    output_dir = Path("./axolotl-output") / run_name  # Create specific output dir
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    yaml_config['output_dir'] = str(output_dir.resolve())

    # Wandb name
    yaml_config['wandb_name'] = run_name

    # Model/Tokenizer specific adjustments
    # Add more model-specific checks if needed
    if "Llama-3" in yaml_config['base_model']:
        yaml_config['chat_template'] = "llama3"
        yaml_config['model_type'] = "LlamaForCausalLM"
        yaml_config['tokenizer_type'] = "LlamaTokenizerFast"
        yaml_config['special_tokens']['pad_token'] = "<|end_of_text|>"
    elif "mistral" in yaml_config['base_model'].lower():
        yaml_config['model_type'] = "AutoModelForCausalLM"  # Use AutoModelForCausalLM for Mistral
        yaml_config['tokenizer_type'] = "AutoTokenizer"  # Use AutoTokenizer for Mistral
        yaml_config['special_tokens']['pad_token'] = "<|end_of_text|>"


    # Add any other specific overrides from run_config here if needed
    # e.g., lora_alpha, lora_dropout, sequence_len etc.
    yaml_config['lora_alpha'] = run_config.get('lora_alpha', yaml_config.get('lora_alpha',
                                                                             16))  # Get from json, or yaml template, or default 16


    # --- Dataset format adjustments (Example) ---
    # You might want to allow overriding dataset format fields via JSON too
    # Example: allow overriding 'field_instruction' from JSON
    # dataset_format_overrides = run_config.get('dataset_format', {})
    # if 'field_instruction' in dataset_format_overrides:
    #    yaml_config['datasets'][0]['type']['field_instruction'] = dataset_format_overrides['field_instruction']
    # if 'format' in dataset_format_overrides:
    #    yaml_config['datasets'][0]['type']['format'] = dataset_format_overrides['format']
    # ... etc for other format fields ...

    print("*****FINAL CONFIG******\n", yaml_config, "\n", "*" * 10)
    return yaml_config


def save_yaml_config(yaml_config, run_name):
    """Saves the generated YAML dictionary to a file."""
    # Create a configs subdirectory if it doesn't exist
    configs_dir = Path("./axolotl-run-configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    yaml_file_path = configs_dir / f"{run_name}.yml"

    try:
        with open(yaml_file_path, 'w') as f:
            yaml.dump(yaml_config, f, sort_keys=False, default_flow_style=False)
        print(f"Generated Axolotl config file: {yaml_file_path}")
        return str(yaml_file_path.resolve())
    except Exception as e:
        print(f"Error saving YAML file {yaml_file_path}: {e}")
        sys.exit(1)


def run_axolotl(yaml_file_path):
    """Runs the Axolotl training command using accelerate."""
    # Ensure the deepspeed config path in the yaml is valid relative to execution context
    # Or use absolute paths in the template/generation step

    command = [
        "axolotl", "train", yaml_file_path
    ]
    print("-" * 50)
    print(f"Executing command: {' '.join(command)}")
    print("-" * 50)

    try:
        # Use subprocess.run to execute the command
        # stream output directly to console
        process = subprocess.run(command, check=True)
        print("-" * 50)
        print("Axolotl training process finished.")
        if process.returncode != 0:
            print(f"Axolotl training failed with return code {process.returncode}")
            sys.exit(process.returncode)

    except FileNotFoundError:
        print("Error: 'accelerate' command not found.")
        print("Please ensure you are in the correct Python environment and accelerate is installed.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during Axolotl execution: {e}")
        # The error message from Axolotl/Accelerate should be visible above
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate Axolotl config and launch fine-tuning.")
    parser.add_argument("run_name", help="The name of the run (key from the JSON config file).")
    parser.add_argument(
        "--config-json",
        default="finetune_configs.json",
        help="Path to the JSON configuration file (default: finetune_configs.json)"
    )
    args = parser.parse_args()

    all_configs = load_config(args.config_json)

    if args.run_name not in all_configs:
        print(f"Error: Run name '{args.run_name}' not found in {args.config_json}")
        print("Available runs:", list(all_configs.keys()))
        sys.exit(1)
    run_config = all_configs[args.run_name]
    print(f"Loaded configuration for run: {args.run_name}")

    axolotl_yaml_data = generate_axolotl_yaml(run_config, args.run_name, AXOLOTL_YAML_TEMPLATE)

    yaml_file_path = save_yaml_config(axolotl_yaml_data, args.run_name)

    # Launch Axolotl using the generated YAML file
    run_axolotl(yaml_file_path)


if __name__ == "__main__":
    main()
