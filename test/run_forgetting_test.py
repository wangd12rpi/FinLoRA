"""
Benchmark every PEFT adapter in ../lora_adapters/ on MMLU and
measure catastrophic forgetting 
"""

from pathlib import Path
from typing import List, Dict
import re, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = Path("mmlu_results.csv")

# "finer", "formula", "headline", "ner", "xbrl_term"
FIN_TASKS = [
    "sentiment", "finer", "xbrl_extract"
    "financebench", "regulations",
]
LORA_SUFFIXES = [
    "_llama_3_1_8b_8bits_r8",
    "_llama_3_1_8b_4bits_r4",
    "_llama_3_1_8b_8bits_r8_dora",
    "_llama_3_1_8b_8bits_r8_rslora",
]
WANTED = {f"{p}{s}" for p in FIN_TASKS for s in LORA_SUFFIXES}

class HFLLM(DeepEvalBaseLLM):
    class _Dummy:
        def __init__(self, ans: str):
            self.answer = ans
    LETTER_RE = re.compile(r"\b([A-E])\b")
    def __init__(self, model, tokenizer, name):
        self._model = model
        self._tokenizer = tokenizer
        self._name = name
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    def load_model(self):
        return self._model
    def _extract_letter(self, text: str) -> str:
        m = self.LETTER_RE.search(text.upper())
        return m.group(1) if m else text.strip()[:1].upper()
    def generate(self, prompt: str, **_) -> str:
        inputs = self._tokenizer([prompt], return_tensors="pt").to(TEST_DEVICE)
        gen_ids = self._model.generate(**inputs, max_new_tokens=10)
        decoded = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return self._extract_letter(decoded)
    async def a_generate(self, prompt: str, **_) -> str:
        return self.generate(prompt)
    def batch_generate(self, prompts: List[str], **_) -> List[List["_Dummy"]]:
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(TEST_DEVICE)
        gen_ids = self._model.generate(**inputs, max_new_tokens=10)
        decoded = [self._tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
        letters = [self._extract_letter(t) for t in decoded]
        return [[self._Dummy(ltr)] for ltr in letters]
    def get_model_name(self):
        return self._name

def eval_on_mmlu(llm_name: str, model) -> float:
    print(f"\n>> Running MMLU for {llm_name}")
    benchmark = MMLU(n_shots=0)
    benchmark.evaluate(model=model, batch_size=4)
    return float(benchmark.overall_score)

def append_row(row: Dict, header: bool = False):
    pd.DataFrame([row]).to_csv(
        CSV_PATH, mode="a" if CSV_PATH.exists() else "w", index=False,
        header=header or not CSV_PATH.exists()
    )

def main():
    finetuned_root = Path("../lora_adapters").resolve()
    def sort_key(p: Path):
        n = p.name
        for t_idx, task in enumerate(FIN_TASKS):
            if n.startswith(task):
                return (t_idx, LORA_SUFFIXES.index(n[len(task):]))
        return (len(FIN_TASKS), 0)
    adapter_dirs = sorted(
        [p for p in finetuned_root.rglob("*") if p.is_dir() and p.name in WANTED],
        key=sort_key
    )
    missing = WANTED - {p.name for p in adapter_dirs}
    if missing:
        print("WARNING: missing adapters:", ", ".join(sorted(missing)))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    print("===== BASE MODEL =====")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
    base.to(TEST_DEVICE)
    base_llm = HFLLM(base, tokenizer, "base")
    base_score = eval_on_mmlu("base", base_llm)
    append_row({"model": "base", "overall": base_score, "Δ% vs base": 0.0}, header=True)
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
        append_row({"model": name, "overall": score, "Δ% vs base": delta_pct})
        del model; del llm
    print("\nSaved →", CSV_PATH)
    print(pd.read_csv(CSV_PATH))

if __name__ == "__main__":
    main()
