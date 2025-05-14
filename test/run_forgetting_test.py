#!/usr/bin/env python
"""
Benchmark every PEFT adapter in ../finetuned_models/ on MMLU and
measure catastrophic forgetting 
"""

from pathlib import Path
from typing import List, Dict

import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# wrap HF model for DeepEval
class HFLLM(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, name):
        self._model = model
        self._tokenizer = tokenizer
        self._name = name

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        inputs = self._tok([prompt], return_tensors="pt").to(TEST_DEVICE)
        gen_ids = self._model.generate(**inputs, max_new_tokens=10)
        return self._tok.decode(gen_ids[0], skip_special_tokens=True)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        inputs = self._tok(prompts, return_tensors="pt", padding=True).to(TEST_DEVICE)
        gen_ids = self._model.generate(**inputs, max_new_tokens=10)
        return [self._tok.decode(g, skip_special_tokens=True) for g in gen_ids]

    def get_model_name(self):
        return self._name

# helper to run one model
def eval_on_mmlu(llm_name: str, model) -> float:
    print(f"\n>> Running MMLU for {llm_name}")
    benchmark = MMLU(n_shots=0)
    benchmark.evaluate(model=model, batch_size=4)
    return float(benchmark.overall_score)   # 0…1

# main experiment
def main():
    finetuned_root = Path("../finetuned_models").resolve()
    adapter_dirs = sorted([p for p in finetuned_root.iterdir() if p.is_dir()])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # baseline (no adapters)
    print("===== BASE MODEL =====")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
    base.to(TEST_DEVICE)
    base_llm = HFLLM(base, tokenizer, "base")
    base_score = eval_on_mmlu("base", base_llm)

    results: List[Dict] = [{"model": "base", "overall": base_score, "Δ% vs base": 0.0}]

    # iterate over every adapter
    for adapter in adapter_dirs:
        torch.cuda.empty_cache()
        name = adapter.name
        print(f"===== {name} =====")

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(adapter))
        model.to(TEST_DEVICE)

        llm = HFLLM(model, tokenizer, name)
        score = eval_on_mmlu(name, llm)
        delta_pct = 100 * (score - base_score) / base_score
        results.append({"model": name, "overall": score, "Δ% vs base": delta_pct})

        # free VRAM before next adapter
        del model; del llm

    df = pd.DataFrame(results)
    df.to_csv("mmlu_results.csv", index=False)
    print("\nSaved → mmlu_results.csv")
    print(df)

if __name__ == "__main__":
    main()
