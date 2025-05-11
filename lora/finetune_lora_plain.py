import json
import yaml
import subprocess
import argparse
import sys
from pathlib import Path

AXOLOTL_YAML_TEMPLATE = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "model_type": "LlamaForCausalLM",
    "tokenizer_type": "AutoTokenizer",

    # Training core
    "gradient_accumulation_steps": 4,
    "micro_batch_size": 2,
    "num_epochs": 4,
    "learning_rate": 2e-4,
    "optimizer": "adamw_torch_fused",
    "lr_scheduler": "cosine",

    # Plain LoRA settings
    "load_in_8bit": False,
    "load_in_4bit": False,
    "adapter": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj"],

    # Dataset placeholder (will be overridden)
    "datasets": [{
        "path": "../data/train/placeholder.jsonl",
        "type": {
            "field_instruction": "context",
            "field_output": "target",
            "format": "[INST] {instruction} [/INST]",
            "no_input_format": "[INST] {instruction} [/INST]"
        }
    }],

    # Misc
    "val_set_size": 0.02,
    "output_dir": "./axolotl-output/placeholder",
    "sequence_len": 4096,
    "gradient_checkpointing": True,
    "logging_steps": 500,
    "warmup_steps": 10,
    "evals_per_epoch": 4,
    "saves_per_epoch": 1,
    "weight_decay": 0.0,
    "special_tokens": {"pad_token": "<|end_of_text|>"},
    "deepspeed": "deepspeed_configs/zero1.json",
    "bf16": "auto",
    "tf32": False
}

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def build_yaml(run_cfg: dict, run_name: str):
    cfg = AXOLOTL_YAML_TEMPLATE.copy()

    cfg["base_model"] = run_cfg["base_model"]
    cfg["lora_r"] = run_cfg.get("lora_r", cfg["lora_r"])
    cfg["learning_rate"] = run_cfg.get("learning_rate", cfg["learning_rate"])
    cfg["num_epochs"] = run_cfg.get("num_epochs", cfg["num_epochs"])
    cfg["gradient_accumulation_steps"] = run_cfg.get(
        "gradient_accumulation_steps", cfg["gradient_accumulation_steps"]
    )
    cfg["micro_batch_size"] = run_cfg.get("batch_size", cfg["micro_batch_size"])

    # dataset path
    data_path = Path(run_cfg["dataset_path"]).resolve()
    cfg["datasets"][0]["path"] = str(data_path)

    # output dir
    out_dir = Path("./axolotl-output") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir.resolve())

    # model‑specific tweaks
    if "Llama-3" in cfg["base_model"]:
        cfg["chat_template"] = "llama3"

    cfg["wandb_name"] = run_name
    return cfg

def save_yaml(cfg: dict, run_name: str) -> str:
    cfg_dir = Path("./axolotl-run-configs")
    cfg_dir.mkdir(exist_ok=True)
    yml_path = cfg_dir / f"{run_name}.yml"
    with open(yml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return str(yml_path)

def run_axolotl(yaml_path: str):
    subprocess.run(["axolotl", "train", yaml_path], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Key from finetune_configs_plain.json")
    parser.add_argument("--config-json", default="finetune_configs_plain.json")
    args = parser.parse_args()

    all_runs = load_json(args.config_json)
    if args.run_name not in all_runs:
        print(f"Run '{args.run_name}' not found.")
        sys.exit(1)

    run_cfg = all_runs[args.run_name]
    yaml_cfg = build_yaml(run_cfg, args.run_name)
    yaml_path = save_yaml(yaml_cfg, args.run_name)
    run_axolotl(yaml_path)

if __name__ == "__main__":
    main()
