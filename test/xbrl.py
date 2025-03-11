import time
import warnings
import inference
import sklearn

warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd

dataset_path = {
    "xbrl_tags_extract": "../data/test/xbrl_xbrl_tags_test.jsonl",
    "xbrl_value_extract": "../data/test/xbrl_value_test.jsonl",
    "xbrl_finer": "../data/test/finer_test_batched.jsonl",
    "xbrl_fnxl": "../data/test/fnxl_test_batched.jsonl",

}


def evaluate_accuracy(out, target):
    correct_count = 0
    response = []
    for x, y in zip(out, target):
        if y in x:
            correct_count += 1
            response.append(y)
        else:
            response.append(x)

    accuracy = correct_count / len(out)
    return accuracy, response


def process_batched(out_text_list, target_list):
    processed_out_text_list = []
    processed_target_list = []

    for out_text, target in zip(out_text_list, target_list):
        split_output = [x.strip().replace("\n", "") for x in out_text.split(',')]  # Split and strip whitespace
        split_target = [x.strip().replace("\n", "") for x in target.split(',')]  # Split and strip whitespace
        processed_target_list += (split_target)  # Keep the split target
        output_len = len(split_output)
        target_len = len(split_target)

        if output_len != target_len:
            if output_len > target_len:
                # Output is longer, truncate
                processed_out_text_list += (split_output[:target_len])
            else:
                # Target is longer, pad output with empty strings
                padding_needed = target_len - output_len
                processed_out_text_list += (split_output + [""] * padding_needed)
        else:
            # Lengths match, use output as is
            processed_out_text_list += (split_output)
    assert len(processed_out_text_list) == len(processed_target_list)
    return processed_out_text_list, processed_target_list


def test_xbrl_tasks(args, dataset_names="xbrl_finer,", prompt_fun=None):
    batch_size = 1
    results = []
    for data_name in dataset_names.split(","):
        if data_name in dataset_path:

            instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
            instructions = instructions
            instructions = instructions.sample(frac=.01, random_state=42)
            # instructions = instructions.sample(2, random_state=42)

            # print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")
            context = instructions['context'].tolist()

            total_steps = instructions.shape[0] // batch_size
            print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

            out_text_list = []

            for i in tqdm(range(total_steps)):
                tmp_context = context[i * batch_size: min(len(context), (i + 1) * batch_size)]

                # tmp_context = ["Here are all USGAAP tags:" + labels + "\n" + x for x in tmp_context]
                # tmp_context = [x + "Answer:" for x in tmp_context]
                # tmp_context = [utils.add_xml(x, limit=80000) for x in tmp_context]

                tmp_target = instructions['target'].tolist()[i * batch_size: min(len(context), (i + 1) * batch_size)]

                out_text = inference.inference(args, tmp_context, if_print_out=True, max_new_token=3000)
                out_text_list += out_text
                print(out_text)
                time.sleep(3)  # avoid rate limit

            instructions["target"] = instructions["target"]
            target_list = instructions["target"].tolist()
            target_list = [str(x) for x in target_list]

            out_text_list, target_list = process_batched(out_text_list, target_list)
            # print(target_list, "\n", out_text_list)

            acc, response = evaluate_accuracy(out_text_list, target_list)

            print(f"Acc {acc}")
            f1 = sklearn.metrics.f1_score(target_list, response, average='weighted')
            # f1 = -1
            print(f"{data_name} Acc: {round(acc * 100, 2)} %. F1: {round(f1, 3)} ")

            results += [{"acc": acc, "f1": f1}]

    return results


if __name__ == '__main__':
    pass
