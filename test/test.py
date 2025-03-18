import torch
import argparse

# from convfinqa import test_convfinqa
from xbrl import test_xbrl_tasks

import sys

sys.path.append('../')


def main(args):


    with torch.no_grad():
        for data in args.dataset.split(','):
            print(data)
            # if data == 'fpb':
            #     test_fpb(args, model, tokenizer)
            # elif data == 'fiqa':
            #     test_fiqa(args, model, tokenizer)
            # elif data == 'tfns':
            #     test_tfns(args, model, tokenizer)
            # elif data == 'nwgi':
            #     test_nwgi(args, model, tokenizer)
            # elif data == 'headline':
            #     test_headline(args, model, tokenizer)
            # elif data == 'ner':
            #     test_ner(args, model, tokenizer)
            if "xbrl" in data:
                test_xbrl_tasks(args, dataset_names=data)
            else:
                raise ValueError('undefined dataset.')

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

    args = parser.parse_args()

    print(args.base_model)
    print(args.peft_model)

    main(args)
