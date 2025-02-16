import json
import random
from os import path


def process_multitask(split, *files):
    complete_data = []
    for i, file in enumerate(files):
        with open(path.join(split, file), 'r') as f:
            for line in f:
                entry = json.loads(line)
                new_entry = {"instruction": entry["context"], "input": "", "output": entry["target"], "task_type": i}
                complete_data.append(new_entry)

    random.shuffle(complete_data)

    with (open(path.join(split,
                         "".join(files).replace("/",
                                                "multi_" + "_".replace(".jsonl", "") + ".jsonl")), 'w')
          as writer):
        for x in complete_data:
            writer.write(json.dumps(x) + "\n")


if __name__ == '__main__':
    process_multitask("train", "finer_train.jsonl", "fingpt_ner_cls_train.jsonl")
