#!/usr/bin/env python
import torch, argparse, sys
sys.path.append("../")
from test_dataset import test_fin_tasks

def main(a):
    with torch.no_grad():
        for d in a.dataset.split(','):
            print("testing:", d)
            test_fin_tasks(a, data_name=d)
    print("Evaluation Ends.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--base_model", default="")
    p.add_argument("--peft_model", default="")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--instruct_template", default="default")
    p.add_argument("--from_remote", action="store_true")
    p.add_argument("--quant_bits", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--source", required=True,
                   choices=["hf", "openai", "together", "anthropic"])
    p.add_argument("--if_print", action="store_true")
    p.add_argument("--sample_ratio", type=float, default=1.0)
    p.add_argument("--model_name", default="")
    p.add_argument("--together_api_key", default="")
    main(p.parse_args())
