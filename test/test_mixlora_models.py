"""
Driver for evaluating MixLoRA adapters on financial benchmarks.
TUDB-Labs/alpaca-mixlora-7b (Llama-2-7B base).
"""
import os, sys, json, argparse, torch, pandas as pd, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

try:
    from mixlora import MixLoraModelForCausalLM
except ImportError:
    sys.exit("MixLoRA missing. Run: pip install mixlora==0.2.3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = {
    "ner": "../data/test/ner_test.jsonl",
    "finer": "../data/test/finer_test_batched.jsonl",
    "fnxl": "../data/test/fnxl_test_batched.jsonl",
}

max_new_token_dict = {
    "ner": "",
    "finer": "",
    "fnxl": "",
}

def load_model(lora_path: str, base_model: str):
    print(f"Loading MixLoRA adapter : {lora_path}")
    print(f"Loading base model      : {base_model}")

    model, _ = MixLoraModelForCausalLM.from_pretrained(
        lora_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok

def batched_generate(model, tok, prompts, max_new=30, batch=8):
    out = []
    for i in range(0, len(prompts), batch):
        batch_prompts = prompts[i:i+batch]
        enc = tok(batch_prompts, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out.extend([tok.decode(g, skip_special_tokens=True) for g in gen])
    
    return out

def save_df(df, tag, src):
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"mixlora_{tag}_{os.path.basename(src).replace('.jsonl','.csv')}"
    )
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

def process_batched(out_text_list, target_list):
    """Process batched outputs for FINER and FNXL"""
    processed_out_text_list = []
    processed_target_list = []

    for out_text, target in zip(out_text_list, target_list):
        split_output = [x.strip().replace("\n", "") for x in out_text.split(',')]
        split_target = [x.strip().replace("\n", "") for x in target.split(',')]
        processed_target_list += (split_target)
        output_len = len(split_output)
        target_len = len(split_target)

        if output_len != target_len:
            if output_len > target_len:
                processed_out_text_list += (split_output[:target_len])
            else:
                padding_needed = target_len - output_len
                processed_out_text_list += (split_output + [""] * padding_needed)
        else:
            processed_out_text_list += (split_output)
    
    return processed_out_text_list, processed_target_list

def evaluate_accuracy(out, target):
    """Evaluate accuracy"""
    correct_count = 0
    response = []
    
    # Normalize outputs and targets
    normalized_out = [x.strip().lower() for x in out]
    normalized_target = [y.strip().lower() for y in target]
    
    for x, y in zip(normalized_out, normalized_target):
        if y in x:
            correct_count += 1
            response.append(y)
        else:
            response.append(x)

    accuracy = correct_count / len(out) if len(out) > 0 else 0
    
    try:
        f1 = f1_score(normalized_target, response, average="weighted", zero_division=0)
    except Exception as e:
        f1 = -1
        print(f"Error calculating F1 score: {e}")
    
    return accuracy, response, f1

def run_task(model, tok, task_name, batch_size, sample_ratio):
    """Run evaluation"""
    print(f"Running {task_name.upper()} evaluation...")
    
    if task_name not in dataset_path:
        print(f"Task {task_name} not found in dataset paths")
        return {"accuracy": 0}
    
    file_path = dataset_path[task_name]
    print(f"Loading data from {file_path}")
    df = pd.read_json(path_or_buf=file_path, lines=True)
    
    if sample_ratio < 1.0:
        sample_size = int(len(df) * sample_ratio)
        df = df.sample(frac=sample_ratio, random_state=42)
    
    context = df['context'].tolist()
    target_list = df['target'].tolist()
    target_list = [str(x) for x in target_list]
    
    total_examples = len(context)
    print(f"Processing {total_examples} examples")
    
    max_tokens = max_new_token_dict.get(task_name, 30)
    
    out_text_list = []
    
    progress_bar = tqdm(total=total_examples, desc=f"Processing {task_name}")
    
    for i in range(0, len(context), batch_size):
        batch_context = context[i:min(len(context), i+batch_size)]
        if not batch_context:
            break
        
        batch_start_time = time.time()
        out_text = batched_generate(model, tok, batch_context, max_new=max_tokens, batch=batch_size)
        out_text_list += out_text
        
        batch_duration = time.time() - batch_start_time
        examples_processed = len(batch_context)
        time_per_example = batch_duration / examples_processed if examples_processed > 0 else 0
        
        progress_bar.set_postfix({"time/example": f"{time_per_example:.2f}s"})
        progress_bar.update(examples_processed)
    
    progress_bar.close()
    
    if task_name in ["finer", "fnxl"]:
        out_text_list, target_list = process_batched(out_text_list, target_list)
    
    acc, response, f1 = evaluate_accuracy(out_text_list, target_list)
    
    print(f"✅ {task_name}: Accuracy: {acc * 100:.2f}%, F1: {f1:.3f}")
    
    results_file = f"{task_name}_mixlora_results.txt"
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    
    with open(os.path.join(SCRIPT_DIR, "results", results_file), "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write(f"Model: alpaca-mixlora-7b\n")
    
    save_df(df, task_name, file_path)
    
    return {"task": task_name, "acc": acc, "f1": f1}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--task_type", required=True,
                   choices=["ner", "finer", "fnxl"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sample_ratio", type=float, default=1.0)
    args = p.parse_args()

    model, tok = load_model(args.model_path, args.base_model)
    
    run_task(model, tok, args.task_type, args.batch_size, args.sample_ratio)

if __name__ == "__main__":
    main() 