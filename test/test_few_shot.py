import os
import gc
import time
import warnings
import sklearn
import torch
import evaluate
import inference
import json
import random
from tqdm import tqdm
import pandas as pd
import argparse
import sys

# allow local imports
sys.path.append(os.path.dirname(__file__))
from test_dataset import process_batched, evaluate_accuracy

# don;'t show warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")

dataset_path = {
    "xbrl_tags_extract":         os.path.join(DATA_DIR, "xbrl_extract_tags_test.jsonl"),
    "xbrl_value_extract":        os.path.join(DATA_DIR, "xbrl_extract_value_test.jsonl"),
    "xbrl_formula_extract":      os.path.join(DATA_DIR, "xbrl_extract_formula_test.jsonl"),
    "xbrl_formula_calc_extract": os.path.join(DATA_DIR, "xbrl_extract_formula_calculations_test.jsonl"),
    "xbrl_finer":                os.path.join(DATA_DIR, "finer_test_batched.jsonl"),
    "xbrl_fnxl":                 os.path.join(DATA_DIR, "fnxl_test_batched.jsonl"),
    "fpb":                       os.path.join(DATA_DIR, "fpb_test.jsonl"),
    "fiqa":                      os.path.join(DATA_DIR, "fiqa_test.jsonl"),
    "tfns":                      os.path.join(DATA_DIR, "tfns_test.jsonl"),
    "nwgi":                      os.path.join(DATA_DIR, "nwgi_test.jsonl"),
    "headline":                  os.path.join(DATA_DIR, "headline_test.jsonl"),
    "ner":                       os.path.join(DATA_DIR, "ner_test.jsonl"),
    "financebench":              os.path.join(DATA_DIR, "financebench_test.jsonl"),
    "xbrl_term":                 os.path.join(DATA_DIR, "xbrl_term_test.jsonl"),
    "formula":                   os.path.join(DATA_DIR, "formula_test.jsonl"),
    "cfa_level1":                os.path.join(DATA_DIR, "cfa_level1_test.jsonl"),
    "cfa_level2":                os.path.join(DATA_DIR, "cfa_level2_test.jsonl"),
    "cfa_level3":                os.path.join(DATA_DIR, "cfa_level3_test.jsonl"),
    "cpa_reg":                   os.path.join(DATA_DIR, "cpa_reg_test.jsonl")
}

train_dataset_path = {
    "xbrl_tags_extract":         os.path.join(TRAIN_DIR, "xbrl_extract_train.jsonl"),
    "xbrl_value_extract":        os.path.join(TRAIN_DIR, "xbrl_extract_train.jsonl"),
    "xbrl_formula_extract":      os.path.join(TRAIN_DIR, "xbrl_extract_train.jsonl"),
    "xbrl_formula_calc_extract": os.path.join(TRAIN_DIR, "xbrl_extract_train.jsonl"),
    "xbrl_finer":                os.path.join(TRAIN_DIR, "finer_train_batched.jsonl"),
    "xbrl_fnxl":                 os.path.join(TRAIN_DIR, "finer_train_batched.jsonl"),
    "fpb":                       os.path.join(TRAIN_DIR, "finlora_sentiment_train.jsonl"),
    "fiqa":                      os.path.join(TRAIN_DIR, "finlora_sentiment_train.jsonl"),
    "tfns":                      os.path.join(TRAIN_DIR, "finlora_sentiment_train.jsonl"),
    "nwgi":                      os.path.join(TRAIN_DIR, "finlora_sentiment_train.jsonl"),
    "headline":                  os.path.join(TRAIN_DIR, "headline_train.jsonl"),
    "ner":                       os.path.join(TRAIN_DIR, "ner_train.jsonl"),
    "financebench":              os.path.join(TRAIN_DIR, "financebench_train.jsonl"),
    "xbrl_term":                 os.path.join(TRAIN_DIR, "xbrl_term_train.jsonl"),
    "formula":                   os.path.join(TRAIN_DIR, "formula_train.jsonl"),
    "cfa_level1":                os.path.join(TRAIN_DIR, "regulations_train.jsonl"),
    "cfa_level2":                os.path.join(TRAIN_DIR, "regulations_train.jsonl"),
    "cfa_level3":                os.path.join(TRAIN_DIR, "regulations_train.jsonl"),
    "cpa_reg":                   os.path.join(TRAIN_DIR, "regulations_train.jsonl")
}

max_new_token_dict = {
    "xbrl_tags_extract": 20,
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
    "financebench": 50,
    "xbrl_term": 50,
    "formula": 50,
    "cfa_level1": 30,
    "cfa_level2": 30,
    "cfa_level3": 30,
    "cpa_reg": 30
}

def load_training_examples(task_name, exclude_context=None):
    """load training examples for few-shot learning, excluding the current test context."""
    train_path = train_dataset_path.get(task_name)
    if not train_path or not os.path.exists(train_path):
        print(f"Warning: No training data found for {task_name}")
        return []
    try:
        with open(train_path, 'r') as f:
            train_examples = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        if 'context' in example and 'target' in example:
                            train_examples.append(example)
                    except json.JSONDecodeError:
                        continue
        if exclude_context:
            filtered_examples = []
            exclude_question = extract_question_from_context(exclude_context)
            for example in train_examples:
                example_question = extract_question_from_context(example['context'])
                if example_question != exclude_question:
                    filtered_examples.append(example)
            return filtered_examples
        return train_examples
    except Exception as e:
        print(f"Error loading training examples for {task_name}: {e}")
        return []

def extract_question_from_context(context):
    """pull out the main question or identifier from a context string."""
    if "Question:" in context:
        question_part = context.split("Question:")[1].split("\n")[0].strip()
        return question_part
    elif "Explain this XBRL term" in context:
        parts = context.split(":")
        if len(parts) >= 2:
            return parts[1].split(".")[0].strip()
    return context[:100]

def construct_few_shot_prompt(test_context, training_examples, num_shots):
    """build a few-shot prompt with input/answer format."""
    if num_shots == 0 or not training_examples:
        return test_context
    selected_examples = random.sample(training_examples, min(num_shots, len(training_examples)))
    few_shot_context = ""
    for i, example in enumerate(selected_examples, 1):
        few_shot_context += f"Example {i}:\nInput: {example['context']}\nAnswer: {example['target']}\n\n"
    few_shot_context += f"Input: {test_context}\nAnswer:"
    return few_shot_context

def test_few_shot_learning(args, data_name, num_shots):
    """run a single dataset with few-shot prompts and standard eval."""
    if data_name not in dataset_path.keys():
        return

    start_time = time.time()
    instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
    sample_size = len(instructions)

    if args.sample_ratio < 1.0:
        sample_size = int(len(instructions) * args.sample_ratio)
        instructions = instructions.sample(frac=args.sample_ratio, random_state=42)

    training_examples = load_training_examples(data_name)
    if not training_examples:
        print(f"No training examples found for {data_name}, skipping {num_shots}-shot")
        return

    model, tokenizer = inference.load_local_model(args)

    task_start_time = time.time()
    context_list = instructions['context'].tolist()
    target_list = instructions["target"].tolist()
    target_list = [str(x) for x in target_list]

    few_shot_contexts = []
    for context in context_list:
        filtered_training = load_training_examples(data_name, exclude_context=context)
        few_shot_context = construct_few_shot_prompt(context, filtered_training, num_shots)
        few_shot_contexts.append(few_shot_context)

    total_steps = len(few_shot_contexts) // args.batch_size + 1
    out_text_list = []

    for i in tqdm(range(total_steps)):
        start_idx = i * args.batch_size
        end_idx = min(len(few_shot_contexts), (i + 1) * args.batch_size)
        tmp_context = few_shot_contexts[start_idx:end_idx]
        if not tmp_context:
            break
        out_text = inference.inference(
            args,
            tmp_context,
            max_new_token=max_new_token_dict.get(data_name, 30),
            model=model,
            tokenizer=tokenizer,
            delimiter="Answer:"
        )
        out_text_list += out_text

    if "finer" in data_name or "fnxl" in data_name:
        out_text_list, target_list = process_batched(out_text_list, target_list)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    per_question_time = (time.time() - task_start_time) / sample_size

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    model_name_clean = (args.model_name or args.base_model).replace("/", "-").replace(":", "-")
    predictions_file = os.path.join(results_dir, f"{data_name}_{model_name_clean}_{num_shots}shot_predictions.json")

    prediction_data = {
        "task": data_name,
        "model": args.model_name or args.base_model,
        "num_shots": num_shots,
        "source": args.source,
        "temperature": args.temperature,
        "sample_size": sample_size,
        "predictions": out_text_list,
        "targets": target_list,
        "contexts": context_list[:len(out_text_list)]
    }

    with open(predictions_file, 'w') as f:
        json.dump(prediction_data, f, indent=2)

    if data_name == "financebench" or data_name == "xbrl_term":
        metric = evaluate.load("bertscore")
        results = metric.compute(predictions=out_text_list, references=target_list, model_type="ProsusAI/finbert")
        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])
        print(
            f"\n✓ {num_shots}-shot {data_name}: precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, Time per question: {per_question_time:.2f}, Batch size: {args.batch_size}"
        )
        return None
    else:
        all_target_type_for_classification = list(set(target_list))
        acc, response = evaluate_accuracy(out_text_list, target_list, all_target_type_for_classification)
        try:
            f1 = sklearn.metrics.f1_score(target_list, response, average='weighted')
        except:
            f1 = -1
            print(f"Error calculating F1 score for {data_name}")
        print(
            f"\n✓ {num_shots}-shot {data_name}: Accuracy: {acc * 100:.3f}%, F1: {f1:.3f}, Time per question: {per_question_time:.2f} s, Batch size: {args.batch_size}"
        )

        results = {"task": data_name, "acc": acc, "f1": f1, "time": per_question_time}

        fname = f"{data_name}_{args.model_name or args.base_model}_{num_shots}shot_results.txt".replace("/", "-")
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, fname), "w+") as f:
            f.write(f"Task: {data_name}\n")
            f.write(f"Accuracy: {acc * 100:.2f}%\n")
            f.write(f"F1 Score: {f1:.3f}\n")
            f.write(f"Per question time: {per_question_time:.2f} minutes\n")
            f.write(f"Model: {args.model_name or args.base_model}\n")
            f.write(f"Shot count: {num_shots}\n")
            f.write(f"Sample Ratio: {args.sample_ratio}\n")
            f.write(f"Temperature: {args.temperature}\n")

        return results

def main(args):
    """run few-shot experiments"""
    # 1, 5
    shot_counts = [3]
    with torch.no_grad():
        for dataset in args.dataset.split(','):
            print(f"testing: {dataset}")
            for num_shots in shot_counts:
                random.seed(42)
                test_few_shot_learning(args, dataset, num_shots)
    print("Evaluation Ends.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot learning evaluation for FinLoRA")
    parser.add_argument("--dataset", required=True, help="Comma-separated list of dataset keys")
    parser.add_argument("--base_model", default="", help="HF model path or remote model name")
    parser.add_argument("--peft_model", default="", help="PEFT adapter path")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument("--from_remote", action="store_true")
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--source", required=True,
                       choices=["hf", "openai", "together", "anthropic", "deepseek", "google"])
    parser.add_argument("--if_print", action="store_true")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--model_name", default="", help="Remote model identifier")
    parser.add_argument("--together_api_key", default="", help="Together AI API key")
    args = parser.parse_args()
    main(args)
