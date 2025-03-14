import time
import warnings
import inference
import sklearn
import sys
import datetime

warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd
import argparse

dataset_path = {
    "xbrl_tags_extract": "../data/test/xbrl_xbrl_tags_test.jsonl",
    "xbrl_value_extract": "../data/test/xbrl_value_test.jsonl",
    "xbrl_formula_extract": "../data/test/xbrl_formula_test.jsonl",
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


def test_xbrl_tasks(args, dataset_names="xbrl_finer,", prompt_fun=None, sample_ratio=1.0):
    start_time = time.time()
    batch_size = 1
    results = []

    print(f"Testing model: {args.base_model} on {dataset_names} with temperature={args.temperature}")

    if not dataset_names or dataset_names.strip() == "all":
        dataset_names = ",".join(dataset_path.keys())

    task_list = dataset_names.split(",")
    total_tasks = len(task_list)
    completed_tasks = 0
    total_examples = 0
    completed_examples = 0

    for data_name in task_list:
        if data_name in dataset_path:
            instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
            if sample_ratio < 1.0:
                sample_size = int(len(instructions) * sample_ratio)
            else:
                sample_size = len(instructions)
            total_examples += sample_size

    main_pbar = tqdm(total=total_examples, position=0, desc="Overall Progress", leave=True)

    for data_name in task_list:
        if data_name in dataset_path:
            task_start_time = time.time()
            completed_tasks += 1

            # Load dataset
            instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
            total_dataset_size = len(instructions)

            # Apply sample ratio to control test size
            if sample_ratio < 1.0:
                instructions = instructions.sample(frac=sample_ratio, random_state=42)

            context = instructions['context'].tolist()
            total_steps = instructions.shape[0] // batch_size
            out_text_list = []

            task_pbar = tqdm(range(total_steps), desc=f"Task {completed_tasks}/{total_tasks}: {data_name}",
                             position=1, leave=False)

            for i in task_pbar:
                example_start_time = time.time()
                completed_examples += 1

                tmp_context = context[i * batch_size: min(len(context), (i + 1) * batch_size)]

                if "extract" in data_name:
                    if "tags" in data_name:
                        tmp_context = [
                            f"Extract all XBRL tags from the following financial document:\n\n{x}\n\nExtracted XBRL tags:"
                            for x in tmp_context]
                    elif "value" in data_name:
                        tmp_context = [
                            f"Extract all XBRL values from the following financial document:\n\n{x}\n\nExtracted XBRL values:"
                            for x in tmp_context]
                    elif "formula" in data_name:
                        tmp_context = [
                            f"Extract the formula from the following financial document:\n\n{x}\n\nExtracted formula:"
                            for x in tmp_context]
                else:
                    if data_name == "xbrl_finer":
                        tmp_context = [
                            f"Tag the following financial document with appropriate XBRL tags (FiNER task):\n\n{x}\n\nTags:"
                            for x in tmp_context]
                    elif data_name == "xbrl_fnxl":
                        tmp_context = [
                            f"Tag the following financial document with appropriate FNXL tags:\n\n{x}\n\nTags:" for x in
                            tmp_context]

                tmp_target = instructions['target'].tolist()[i * batch_size: min(len(context), (i + 1) * batch_size)]

                overall_percent = f"{completed_examples}/{total_examples}"
                task_pbar.set_description(f"Task {data_name} - {overall_percent}")

                out_text = inference.inference(args, tmp_context, if_print_out=False, max_new_token=3000)
                out_text_list += out_text

                main_pbar.update(1)

                # Sleep to avoid rate limiting
                time.sleep(0.1)

            task_pbar.close()
            instructions["target"] = instructions["target"]
            target_list = instructions["target"].tolist()
            target_list = [str(x) for x in target_list]

            if "finer" in data_name or "fnxl" in data_name:
                out_text_list, target_list = process_batched(out_text_list, target_list)

            acc, response = evaluate_accuracy(out_text_list, target_list)

            try:
                f1 = sklearn.metrics.f1_score(target_list, response, average='weighted')
            except:
                f1 = -1
                print(f"Error calculating F1 score for {data_name}")

            task_time = time.time() - task_start_time

            print(f"\n✓ {data_name}: Accuracy: {acc * 100:.2f}%, F1: {f1:.3f}, Time: {task_time / 60:.2f} min")

            results += [{"task": data_name, "acc": acc, "f1": f1, "time": task_time}]

            # Save results to file
            with open(f"{data_name}_results.txt", "w") as f:
                f.write(f"Task: {data_name}\n")
                f.write(f"Accuracy: {acc * 100:.2f}%\n")
                f.write(f"F1 Score: {f1:.3f}\n")
                f.write(f"Time Taken: {task_time / 60:.2f} minutes\n")
                f.write(f"Model: {args.base_model}\n")
                f.write(f"Sample Ratio: {sample_ratio}\n")
                f.write(f"Temperature: {args.temperature}\n")

    main_pbar.close()

    # Calculate total time taken
    total_time = time.time() - start_time

    print(f"\nComplete: {total_time / 60:.2f} minutes total")
    for result in results:
        print(f"- {result['task']}: Acc: {result['acc'] * 100:.2f}%, F1: {result['f1']:.3f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", type=str,
                        help="XBRL tasks to test, comma-separated. Use 'all' for all tasks.")
    parser.add_argument("--base_model", default="together/deepseek-v3", type=str, help="Model to test")
    parser.add_argument("--peft_model", required=False, default="", type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--quant_bits", default=8, type=int)
    parser.add_argument("--sample_ratio", default=0.01, type=float, help="Ratio of data to sample for testing")
    parser.add_argument("--together_api_key", required=True, type=str, help="API key for Together AI")
    parser.add_argument("--temperature", default=0.0, type=float, help="Temperature for text generation")

    args = parser.parse_args()

    # Run XBRL tasks test
    results = test_xbrl_tasks(args, dataset_names=args.dataset, sample_ratio=args.sample_ratio)
