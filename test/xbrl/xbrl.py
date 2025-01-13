import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

dataset_path = {
    "xbrl_tags": "../xbrl/xbrl_xbrl_tags_test.jsonl",
    "xbrl_finer": "../data/test/finer_test.jsonl",
}


def evaluate_accuracy(out, target):
    correct_count = 0
    for x, y in zip(out, target):
        if y in x:
            correct_count += 1

    accuracy = correct_count / len(out)
    return accuracy


def test_xbrl(args, model, tokenizer, path="finer,", prompt_fun=None):
    batch_size = 128
    results = []
    for data_name in path.split(","):
        if data_name in dataset_path:

            instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
            instructions = instructions
            # instructions = instructions.head(10)
            # print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")
            context = instructions['context'].tolist()

            total_steps = instructions.shape[0] // batch_size
            print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

            out_text_list = []

            for i in tqdm(range(total_steps)):
                tmp_context = context[i * batch_size: min(len(context), (i + 1) * batch_size)]
                tmp_context = [x + "Answer:" for x in tmp_context]
                # tmp_context = [utils.add_xml(x, limit=80000) for x in tmp_context]
                tmp_target = instructions['target'].tolist()[i * batch_size: min(len(context), (i + 1) * batch_size)]

                if len(tmp_context) == 0:
                    continue
                tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512,
                                   return_token_type_ids=False)
                for k in tokens.keys():
                    tokens[k] = tokens[k].cuda()
                res = model.generate(**tokens, max_new_tokens=60, eos_token_id=tokenizer.eos_token_id)
                res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
                out_text = ["".join(o.split("\n")[1:]).strip() for o in res_sentences]
                # print(out_text)

                out_text_list += out_text
                # torch.cuda.empty_cache()

            instructions["target"] = instructions["target"]
            target_list = instructions["target"].tolist()
            target_list = [str(x) for x in target_list]

            acc = evaluate_accuracy(out_text_list, target_list)
            print(f"{data_name} Acc: {acc}. ")

            results += [{"acc": acc, "f1": -1}]

    return results
