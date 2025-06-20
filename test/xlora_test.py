import argparse, json, os, sys, warnings, torch, xlora
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

TASK_TO_TESTDATA = {
    "finer": "data/test/finer_test.jsonl",
    "ner": "data/test/ner_test.jsonl",
    "headline": "data/test/headline_test.jsonl",
    "fiqa": "data/test/fiqa_test.jsonl",
    "fpb": "data/test/fpb_test.jsonl",
    "tfns": "data/test/tfns_test.jsonl",
}

TASK_TO_ADAPTER = {
    "finer": "lora_adapters/8bits-r8/finer_train_batched.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
    "ner": "lora_adapters/8bits-r8/fingpt_ner_cls_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
    "headline": "lora_adapters/8bits-r8/fingpt_headline_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
    "fiqa": "lora_adapters/8bits-r8/fingpt_headline_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
    "fpb": "lora_adapters/8bits-r8/fingpt_headline_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
    "tfns": "lora_adapters/8bits-r8/fingpt_headline_train.jsonl-meta-llama-Llama-3.1-8B-Instruct-8bits-r8",
}


def load_xlora_model(base_model: str):
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        config=config,
    )

    adapters = {
        t: p for t, p in TASK_TO_ADAPTER.items() if os.path.exists(p)
    }
    if not adapters:
        raise RuntimeError("No valid adapters found")

    xlora_cfg = xlora.xLoRAConfig(
        base_model_id=base_model,
        adapters=adapters,
        hidden_size=config.hidden_size,
        xlora_depth=4,
        device=torch.device(device),
    )
    model = xlora.add_xlora_to_model(model=model, xlora_config=xlora_cfg)
    model.eval()

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    return model, tok


def format_prompt(task, example):
    text = example.get("context", "")
    if task == "finer":
        return f"Extract financial entities:\n\n{text}\n\nEntities:"
    if task == "ner":
        return f"Find named entities:\n\n{text}\n\nEntities:"
    if task == "headline":
        return f"Generate a headline:\n\n{text}\n\nHeadline:"
    if task == "fiqa":
        return f"Answer:\n\n{text}\n\nAnswer:"
    if task in ("fpb", "tfns"):
        return f"Classify sentiment:\n\n{text}\n\nSentiment:"
    return text


def run_inference(model, tok, prompts, temperature=0.0, top_p=1.0):
    outs = []
    for p in prompts:
        enc = {k: v.to(device) for k, v in tok(p, return_tensors="pt").items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=False,
            )
        text = tok.decode(gen[0], skip_special_tokens=True)
        outs.append(text[len(p):].strip() if text.startswith(p) else text)
    return outs


def parse_output(task, out):
    if task in ("tfns", "fpb"):
        low = out.lower()
        for lbl in ("positive", "negative", "neutral"):
            if lbl in low:
                return lbl
    return out.strip()


def evaluate(task, preds, gold):
    if task in ("tfns", "fpb"):
        return {
            "accuracy": accuracy_score(gold, preds),
            "f1": f1_score(gold, preds, average="weighted", zero_division=0),
        }
    acc = sum(p.strip() == g.strip() for p, g in zip(preds, gold)) / len(preds)
    return {"accuracy": acc, "f1": 0.0}


def test_xlora(args):
    model, tok = load_xlora_model(args.base_model)
    results, tasks = {}, (
        TASK_TO_TESTDATA.keys() if args.tasks == "all" else args.tasks.split(",")
    )

    for task in tasks:
        path = TASK_TO_TESTDATA.get(task)
        if not path or not os.path.exists(path):
            print(f"Skipped {task}")
            continue

        data = [json.loads(l) for l in open(path)]
        data = data[: args.max_samples or 50]
        prompts = [format_prompt(task, ex) for ex in data]

        preds = []
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            preds += run_inference(
                model,
                tok,
                prompts[i : i + args.batch_size],
                args.temperature,
                args.top_p,
            )

        parsed = [parse_output(task, o) for o in preds]
        gold = [ex.get("target", "") for ex in data]
        metrics = evaluate(task, parsed, gold)

        results[task] = {"metrics": metrics, "examples": parsed[:5]}
        print(f"{task}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    os.makedirs("test/results", exist_ok=True)
    json.dump(results, open("test/results/xlora_results.json", "w"), indent=2)
    print("Saved to test/results/xlora_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tasks", default="all")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=50)
    args = p.parse_args()

    print(vars(args))
    test_xlora(args)