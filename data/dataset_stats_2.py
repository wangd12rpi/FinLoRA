#!/usr/bin/env python
# embed_sim_finance_test19.py
#
# Build a 19×19 cosine‑similarity matrix (context + target)
# for the full financial‑benchmark *test* split.
# ----------------------------------------------------------------------

import os, json, random, itertools, torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --------------------------- CONFIG -----------------------------------
BASE_DIR    = "test"        # change if the test files live elsewhere
MAX_LINES   = 200           # sample cap per dataset
RANDOM_SEED = 42
MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
SEP         = " <SEP> "

random.seed(RANDOM_SEED)
torch.set_grad_enabled(False)

FILES = {  # 19 test sets
    "cfa_level1_test.jsonl":                         "CFA‑L1",
    "cfa_level2_test.jsonl":                         "CFA‑L2",
    "cfa_level3_test.jsonl":                         "CFA‑L3",
    "cpa_reg_test.jsonl":                            "CPA",
    "financebench_test.jsonl":                       "FinanceBench",
    "finer_test_batched.jsonl":                      "FiNER",
    "fiqa_test.jsonl":                               "FiQA",
    "fnxl_test_batched.jsonl":                       "FNXL",
    "formula_test.jsonl":                            "FinancialMath",
    "fpb_test.jsonl":                                "FPB",
    "headline_test.jsonl":                           "Headline",
    "ner_test.jsonl":                                "NER",
    "nwgi_test.jsonl":                               "NWGI",
    "tfns_test.jsonl":                               "TFNS",
    "xbrl_extract_formula_calculations_test.jsonl":  "XBRL‑Calc",
    "xbrl_extract_formula_test.jsonl":               "XBRL‑Form",
    "xbrl_extract_tags_test.jsonl":                  "XBRL‑Tag",
    "xbrl_extract_value_test.jsonl":                 "XBRL‑Val",
    "xbrl_term_test.jsonl":                          "XBRL‑Term"
}
# ----------------------------------------------------------------------

def load_concat(path, cap=MAX_LINES):
    """Return list of 'context SEP target' strings."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rows.append(f"{rec.get('context','').strip()}{SEP}{rec.get('target','').strip()}")
            except json.JSONDecodeError:
                continue
    return random.sample(rows, cap) if len(rows) > cap else rows

def centroid(samples, model):
    """Mean embedding vector for a list of strings."""
    if not samples:
        return torch.zeros(model.get_sentence_embedding_dimension())
    emb = model.encode(samples, convert_to_tensor=True,
                       batch_size=32, show_progress_bar=False)
    return emb.mean(dim=0)

def build_full_matrix(file_map, model):
    """Return DataFrame with cosine similarities for all datasets."""
    cents = {short: centroid(load_concat(os.path.join(BASE_DIR, fn)), model)
             for fn, short in file_map.items()}
    names = list(cents)
    df = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, j in itertools.combinations_with_replacement(names, 2):
        sim = util.cos_sim(cents[i], cents[j]).item()
        df.at[i, j] = df.at[j, i] = round(sim, 3)
    return df

def main():
    model = SentenceTransformer(MODEL_NAME, device="cpu")  # use 'cuda' if you have a GPU
    df = build_full_matrix(FILES, model)
    print("### 19 × 19 Similarity Matrix (context + target)\n")
    print(df.to_markdown())
    print()

if __name__ == "__main__":
    main()
