import torch
import argparse
import sys
import os

sys.path.append("../")
from test_dataset import test_fin_tasks

def main(args):
    with torch.no_grad():
        for dataset in args.dataset.split(','):
            print("testing:", dataset)
            test_fin_tasks(args, data_name=dataset)
    print("Evaluation Ends.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True,               help="Comma-separated list of dataset keys")
    parser.add_argument("--base_model",   default="",                 help="HF model path or remote model name")
    parser.add_argument("--peft_model",   default="",                 help="PEFT adapter path")
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument("--from_remote",  action="store_true")
    parser.add_argument("--quant_bits",   type=int, default=8)
    parser.add_argument("--temperature",  type=float, default=0.0)
    parser.add_argument(
        "--source", required=True,
        choices=["hf", "openai", "together", "anthropic", "deepseek", "google"]
    )
    parser.add_argument("--if_print",     action="store_true")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--model_name",   default="",                 help="Remote model identifier")
    parser.add_argument("--together_api_key", default="",            help="Together AI API key")
    args = parser.parse_args()
    main(args)
