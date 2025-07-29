import os
import json
import random
from pathlib import Path

# Get the script directory
script_dir = Path(__file__).parent.absolute()
data_dir = script_dir
train_dir = data_dir / "train"

# Define the dataset groups
DATASET_GROUPS = {
    "general_multi": ["finlora_sentiment_train.jsonl", "headline_train.jsonl", "ner_train.jsonl"],
    "reporting_multi": ["finer_train_batched.jsonl", "xbrl_term_train.jsonl"],
    "analysis_multi": ["financebench_train.jsonl", "xbrl_extract_train.jsonl", "formula_train.jsonl"]
}

# Create the output directory if it doesn't exist
output_dir = train_dir / "multi"
os.makedirs(output_dir, exist_ok=True)

def load_jsonl(file_path):
    """Load a JSONL file and return a list of its contents."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save a list of JSON objects to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def balance_datasets(datasets, names):
    """Balance datasets according to the requirements."""
    counts = {name: len(data) for name, data in zip(names, datasets)}
    print(f"Original dataset counts: {counts}")

    # Apply balancing logic for each group
    if "finlora_sentiment_train.jsonl" in names and "headline_train.jsonl" in names and "ner_train.jsonl" in names:
        # For general_multi: reduce the size of headline and sentiment to match ner
        ner_idx = names.index("ner_train.jsonl")
        sentiment_idx = names.index("finlora_sentiment_train.jsonl")
        headline_idx = names.index("headline_train.jsonl")

        ner_count = len(datasets[ner_idx])

        # Reduce sentiment and headline to match ner
        datasets[sentiment_idx] = random.sample(datasets[sentiment_idx], ner_count)
        datasets[headline_idx] = random.sample(datasets[headline_idx], ner_count)

    elif "finer_train_batched.jsonl" in names and "xbrl_term_train.jsonl" in names:
        # For reporting_multi: increase the number of xbrl_term
        finer_idx = names.index("finer_train_batched.jsonl")
        xbrl_term_idx = names.index("xbrl_term_train.jsonl")

        finer_count = len(datasets[finer_idx])
        xbrl_term_count = len(datasets[xbrl_term_idx])

        # Calculate how many times to duplicate xbrl_term to roughly match finer
        multiplier = finer_count // xbrl_term_count + 1

        # Duplicate xbrl_term data
        datasets[xbrl_term_idx] = datasets[xbrl_term_idx] * multiplier

        # Trim to match finer count if needed
        if len(datasets[xbrl_term_idx]) > finer_count:
            datasets[xbrl_term_idx] = datasets[xbrl_term_idx][:finer_count]

    elif "financebench_train.jsonl" in names and "formula_train.jsonl" in names and "xbrl_extract_train.jsonl" in names:
        # For analysis_multi: increase the number of financebench and formula
        financebench_idx = names.index("financebench_train.jsonl")
        formula_idx = names.index("formula_train.jsonl")
        xbrl_extract_idx = names.index("xbrl_extract_train.jsonl")

        xbrl_extract_count = len(datasets[xbrl_extract_idx])
        financebench_count = len(datasets[financebench_idx])
        formula_count = len(datasets[formula_idx])

        # Calculate multipliers
        financebench_multiplier = xbrl_extract_count // financebench_count + 1
        formula_multiplier = xbrl_extract_count // formula_count + 1

        # Duplicate data
        datasets[financebench_idx] = datasets[financebench_idx] * financebench_multiplier
        datasets[formula_idx] = datasets[formula_idx] * formula_multiplier

        # Trim to match xbrl_extract count if needed
        if len(datasets[financebench_idx]) > xbrl_extract_count:
            datasets[financebench_idx] = datasets[financebench_idx][:xbrl_extract_count]
        if len(datasets[formula_idx]) > xbrl_extract_count:
            datasets[formula_idx] = datasets[formula_idx][:xbrl_extract_count]

    balanced_counts = {name: len(data) for name, data in zip(names, datasets)}
    print(f"Balanced dataset counts: {balanced_counts}")

    return datasets

def main():
    # Process each dataset group
    for group_name, dataset_files in DATASET_GROUPS.items():
        print(f"\nProcessing {group_name}...")

        # Load all datasets in this group
        datasets = []
        for file_name in dataset_files:
            file_path = train_dir / file_name
            print(f"Loading {file_path}...")
            data = load_jsonl(file_path)
            print(f"Loaded {len(data)} examples from {file_path}")
            datasets.append(data)

        # Balance datasets
        balanced_datasets = balance_datasets(datasets, dataset_files)

        # Merge all datasets in this group
        merged_data = []
        for dataset in balanced_datasets:
            merged_data.extend(dataset)

        # Shuffle the merged data
        random.shuffle(merged_data)

        # Save the merged dataset
        output_path = output_dir / f"{group_name}.jsonl"
        save_jsonl(merged_data, output_path)
        print(f"Saved {len(merged_data)} examples to {output_path}")

if __name__ == "__main__":
    main()
