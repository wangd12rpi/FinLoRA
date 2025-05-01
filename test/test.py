import torch
import argparse

# from convfinqa import test_convfinqa
from test_dataset import test_fin_tasks

import sys

sys.path.append('../')


def main(args):
    with torch.no_grad():
        for data in args.dataset.split(','):
            print("testing:", data)
            test_fin_tasks(args, data_name=data)

    print('Evaluation Ends.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--peft_model", required=False, default="", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int, help="The train batch size per device")
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--quant_bits", default=8, type=int)
    parser.add_argument("--temperature", default=0.0, type=float, help="Temperature for text generation")
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--if_print", required=False, type=bool, default=False)

    args = parser.parse_args()

    print(args.base_model)
    print(args.peft_model)

    main(args)
