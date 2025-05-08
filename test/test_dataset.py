import time
import warnings
import sklearn
from sklearn.metrics import f1_score, accuracy_score
import inference

warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd

dataset_path = {
    "xbrl_tags_extract": "../data/test/xbrl_extract_tags_test.jsonl",
    "xbrl_value_extract": "../data/test/xbrl_extract_value_test.jsonl",
    "xbrl_formula_extract": "../data/test/xbrl_extract_formula_test.jsonl",
    "xbrl_formula_calc_extract": "../data/test/xbrl_extract_formula_calculations_test.jsonl",
    "xbrl_finer": "../data/test/finer_test_batched.jsonl",
    "xbrl_fnxl": "../data/test/fnxl_test_batched.jsonl",
    "fpb": "../data/test/fpb_test.jsonl",
    "fiqa": "../data/test/fiqa_test.jsonl",
    "tfns": "../data/test/tfns_test.jsonl",
    "nwgi": "../data/test/nwgi_test.jsonl",
    "headline": "../data/test/headline_test.jsonl",
    "ner": "../data/test/ner_test.jsonl",
}

max_new_token_dict = {
    "xbrl_tags_extract": 10,
    "xbrl_value_extract": 20,
    "xbrl_formula_extract": 30,
    "xbrl_formula_calc_extract": 30,
    "xbrl_finer": 100,
    "xbrl_fnxl": 100,
    "fpb": 10,
    "fiqa": 10,
    "tfns": 10,
    "nwgi": 10,
    "headline": 10,
    "ner": 10,
}

max_new_token_dict_for_base_models = {
    "xbrl_tags_extract": 60,
    "xbrl_value_extract": 60,
    "xbrl_formula_extract": 60,
    "xbrl_formula_calc_extract": 60,
    "xbrl_finer": 100,
    "xbrl_fnxl": 100,
    "fpb": 60,
    "fiqa": 60,
    "tfns": 60,
    "nwgi": 60,
    "headline": 60,
    "ner": 60,
}


def evaluate_accuracy(out, target, task_name=""):
    correct_count = 0
    response = []
    
    # Normalize outputs and targets - strip whitespace and lowercase
    normalized_out = [x.strip().lower() for x in out]
    normalized_target = [y.strip().lower() for y in target]
    
    for x, y in zip(normalized_out, normalized_target):
        if y in x:
            correct_count += 1
            response.append(y)
        else:
            response.append(x)

    accuracy = correct_count / len(out)
    
    # Calculate F1 score with task-specific handling
    try:
        if task_name in ["fpb", "tfns"]:
            # For classification tasks, use weighted average and handle zero division
            f1 = f1_score(normalized_target, response, average="weighted", zero_division=0)
        elif len(set(normalized_target)) <= 1:
            # If there's only one class in target, F1 calculation will fail
            f1 = 1.0 if accuracy == 1.0 else 0.0
        else:
            # For other tasks, still try weighted F1 but with zero division handling
            f1 = f1_score(normalized_target, response, average="weighted", zero_division=0)
    except Exception as e:
        f1 = -1
        print(f"Error calculating F1 score for {task_name}: {e}")
    
    return accuracy, response, f1


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


def test_fin_tasks(args, data_name="xbrl_finer", prompt_fun=None):
    start_time = time.time()
    batch_size = args.batch_size
    results = {}

    print(f"Testing model: {args.base_model} on {data_name} with temperature={args.temperature}")

    if data_name not in dataset_path.keys():
        return results

    instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
    sample_size = len(instructions)

    if args.sample_ratio < 1.0:
        sample_size = int(len(instructions) * args.sample_ratio)
        instructions = instructions.sample(frac=args.sample_ratio, random_state=42)

    model, tokenizer = inference.load_local_model(args)

    task_start_time = time.time()

    context = instructions['context'].tolist()
    target_list = instructions["target"].tolist()
    target_list = [str(x) for x in target_list]
    
    total_steps = instructions.shape[0] // batch_size
    out_text_list = []

    task_pbar = tqdm(range(total_steps + 1))

    for i in task_pbar:
        tmp_context = context[i * batch_size: min(len(context), (i + 1) * batch_size)]
        if not tmp_context:
            break
        tmp_target = instructions['target'].tolist()[i * batch_size: min(len(context), (i + 1) * batch_size)]
        
        batch_start_time = time.time()
        out_text = inference.inference(args, tmp_context, max_new_token=max_new_token_dict.get(data_name, 30), model=model,
                                       tokenizer=tokenizer)
        batch_duration = time.time() - batch_start_time
        
        out_text_list += out_text
        
        # Update progress bar with time per example
        examples_in_batch = len(tmp_context)
        time_per_example = batch_duration / examples_in_batch if examples_in_batch > 0 else 0
        task_pbar.set_description(f"Processing {data_name}: {time_per_example:.2f}s per example")

    # instructions["target"] = instructions["target"]

    if "finer" in data_name or "fnxl" in data_name:
        out_text_list, target_list = process_batched(out_text_list, target_list)

    acc, response, f1 = evaluate_accuracy(out_text_list, target_list, data_name)

    per_question_time = (time.time() - task_start_time) / sample_size

    print(f"\n✓ {data_name}: Accuracy: {acc * 100:.2f}%, F1: {f1:.3f}, Time per quesiton: {per_question_time:.2f} s")

    results = {"task": data_name, "acc": acc, "f1": f1, "time": per_question_time}

    fname = f"{data_name}_{args.base_model}_{args.peft_model}_results.txt".replace("/", "-")
    # Save results to file
    with open(f"results/{fname}", "w+") as f:
        f.write(f"Task: {data_name}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write(f"Per question time: {per_question_time:.2f} seconds\n")
        f.write(f"Model: {args.base_model}\n")
        f.write(f"PEFT Model: {args.peft_model}\n")
        f.write(f"Sample Ratio: {args.sample_ratio}\n")
        f.write(f"Temperature: {args.temperature}\n")

    return results
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default="all", type=str,
#                         help="XBRL tasks to test, comma-separated. Use 'all' for all tasks.")
#     parser.add_argument("--base_model", default="together/deepseek-v3", type=str, help="Model to test")
#     parser.add_argument("--peft_model", required=False, default="", type=str)
#     parser.add_argument("--max_length", default=4096, type=int)
#     parser.add_argument("--batch_size", default=1, type=int)
#     parser.add_argument("--quant_bits", default=8, type=int)
#     parser.add_argument("--sample_ratio", default=0.01, type=float, help="Ratio of data to sample for testing")
#     parser.add_argument("--together_api_key", required=True, type=str, help="API key for Together AI")
#     parser.add_argument("--temperature", default=0.0, type=float, help="Temperature for text generation")
#
#     args = parser.parse_args()
#
#     # Run XBRL tasks test
#     results = test_xbrl_tasks(args, dataset_names=args.dataset, sample_ratio=args.sample_ratio)
