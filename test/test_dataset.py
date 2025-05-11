import os
import gc
import time
import warnings
import sklearn
import torch
import evaluate
import inference

from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

# Compute data directory relative to this file
BASE_DIR = os.path.dirname(__file__)                     # .../FinLoRA/test
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")   # .../FinLoRA/data/test

# Map task names to their JSONL files in the data/test directory
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
}

# Maximum new tokens per dataset
max_new_token_dict = {
    "xbrl_tags_extract":         20,
    "xbrl_value_extract":        20,
    "xbrl_formula_extract":      30,
    "xbrl_formula_calc_extract": 30,
    "xbrl_finer":                100,
    "xbrl_fnxl":                 100,
    "fpb":                       10,
    "fiqa":                      10,
    "tfns":                      10,
    "nwgi":                      10,
    "headline":                  10,
    "ner":                       10,
    "financebench":              50,
    "xbrl_term":                 50,
    "formula":                   50,
}


def evaluate_accuracy(out, target, target_type_list):
    correct_count = 0
    response = []
    types_lower = [str(t).lower() for t in target_type_list]

    if len(out) != len(target):
        raise ValueError("Input lists 'out' and 'target' must have the same length.")

    for x, y in zip(out, target):
        x_str = str(x).lower()
        y_str = str(y).lower()
        matches = [(lbl, x_str.find(lbl)) for lbl in types_lower if x_str.find(lbl) != -1]
        if matches:
            first = sorted(matches, key=lambda t: t[1])[0][0]
            if first == y_str:
                correct_count += 1
                response.append(y)
                continue
        response.append(x)

    accuracy = correct_count / len(out) if out else 0.0
    return accuracy, response


def process_batched(out_list, tgt_list):
    proc_out, proc_tgt = [], []
    for o, t in zip(out_list, tgt_list):
        so = [x.strip().replace("\n", "") for x in o.split(',')]
        st = [x.strip().replace("\n", "") for x in t.split(',')]
        proc_tgt.extend(st)
        if len(so) != len(st):
            if len(so) > len(st):
                proc_out.extend(so[:len(st)])
            else:
                proc_out.extend(so + [""] * (len(st) - len(so)))
        else:
            proc_out.extend(so)
    assert len(proc_out) == len(proc_tgt)
    return proc_out, proc_tgt


def test_fin_tasks(args, data_name="xbrl_finer", prompt_fun=None):
    print(f"Testing model: {args.base_model} on {data_name} with temperature={args.temperature}")

    if data_name not in dataset_path:
        return {}

    # Load the JSONL instructions
    instructions = pd.read_json(dataset_path[data_name], lines=True)
    if args.sample_ratio < 1.0:
        instructions = instructions.sample(frac=args.sample_ratio, random_state=42)

    model, tokenizer = inference.load_local_model(args)
    contexts = instructions['context'].tolist()
    targets = [str(x) for x in instructions['target'].tolist()]

    out_texts = []
    total = len(contexts)
    for i in tqdm(range(0, total, args.batch_size), desc=f"Processing {data_name}", total=(total + args.batch_size - 1) // args.batch_size):
        batch_ctx = contexts[i:i+args.batch_size]
        out = inference.inference(
            args,
            batch_ctx,
            max_new_token=max_new_token_dict.get(data_name, 30),
            model=model,
            tokenizer=tokenizer,
            delimiter="Answer:"
        )
        out_texts.extend(out)

    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    per_question_time = (time.time() - task_start_time) / sample_size

    if data_name == "financebench" or data_name == "xbrl_term":
        frugal_metric = evaluate.load("bertscore")
        results = frugal_metric.compute(predictions=out_text_list, references=target_list, lang="en",)
        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])
        print(
            f"\n✓ {data_name}: precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, Time per question: {per_question_time:.2f}, Batch size: {batch_size}")
        return None

    # Evaluate
    if data_name in ("financebench", "xbrl_term"):
        metric = evaluate.load("bertscore")
        res = metric.compute(predictions=out_texts, references=targets, lang="en")
        p = sum(res["precision"]) / len(res["precision"])
        r = sum(res["recall"]) / len(res["recall"])
        f1 = sum(res["f1"]) / len(res["f1"])
        print(f"✓ {data_name}: P={p:.2f}, R={r:.2f}, F1={f1:.2f}")
        return {}
    else:
        types = list(set(targets))
        acc, resp = evaluate_accuracy(out_texts, targets, types)
        try:
            f1 = sklearn.metrics.f1_score(targets, resp, average='weighted')
        except Exception:
            f1 = -1
        print(f"✓ {data_name}: Acc={acc*100:.3f}%, F1={f1:.3f}")
        return {"task": data_name, "acc": acc, "f1": f1}
