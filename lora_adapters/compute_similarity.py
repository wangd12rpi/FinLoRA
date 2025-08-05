#!/usr/bin/env python3
"""
Main code to analyze LoRA parameter interference in financial tasks
This generates the similarity scores used in the reviewer response.
"""

import torch
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from collections import defaultdict
import os


def load_adapter(path):
    """Load LoRA adapter and compute effective weight updates"""
    safetensors_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Warning: {safetensors_path} not found")
        return None

    print(f"Loading: {safetensors_path}")
    sd = load_file(safetensors_path, device="cpu")

    layers = defaultdict(dict)

    # Parse LoRA weight keys
    for key, tensor in sd.items():
        if ".lora_A.weight" in key:
            layer_name = key.replace(".lora_A.weight", "")
            layers[layer_name]['A'] = tensor.float()
        elif ".lora_B.weight" in key:
            layer_name = key.replace(".lora_B.weight", "")
            layers[layer_name]['B'] = tensor.float()

    # Compute effective updates: ΔW = α/r * B @ A
    # With α=8, r=8 → scale=1, so ΔW = B @ A
    updates = {}
    for layer_name, matrices in layers.items():
        if 'A' in matrices and 'B' in matrices:
            delta_w = matrices['B'] @ matrices['A']
            updates[layer_name] = delta_w.flatten()

    print(f"  Found {len(updates)} LoRA layers")
    return updates


def cosine_similarity(u, v, eps=1e-8):
    """Compute cosine similarity between two vectors"""
    u, v = u.float(), v.float()
    u_norm, v_norm = torch.norm(u), torch.norm(v)

    if u_norm < eps or v_norm < eps:
        return 0.0

    return float(torch.dot(u, v) / (u_norm * v_norm))


def analyze_interference():
    """Main analysis function - reproduces the evidence in interference_evidence.csv"""

    print("LoRA Parameter Interference Analysis")
    print("=" * 50)

    # Financial task categories based on performance table
    task_configs = {
        # Financial Statement Analysis Tasks (positive transfer)
        'formula': '8bits_r8/formula_llama_3_1_8b_8bits_r8',
        'financebench': '8bits_r8/financebench_llama_3_1_8b_8bits_r8',
        'xbrl_extract': '8bits_r8/xbrl_extract_llama_3_1_8b_8bits_r8',

        # General Financial Tasks (severe interference)  
        'headline': '8bits_r8/headline_llama_3_1_8b_8bits_r8',
        'ner': '8bits_r8/ner_llama_3_1_8b_8bits_r8',
        'sentiment': '8bits_r8/sentiment_llama_3_1_8b_8bits_r8',

        # Financial Reporting Tasks (catastrophic interference)
        'finer': '8bits_r8/finer_llama_3_1_8b_8bits_r8',
        'xbrl_term': '8bits_r8/xbrl_term_llama_3_1_8b_8bits_r8',

        # Financial Certification Tasks
        'regulations': '8bits_r8/regulations_llama_3_1_8b_8bits_r8'
    }

    print("\nLoading LoRA adapters...")
    task_updates = {}
    for task, path in task_configs.items():
        updates = load_adapter(path)
        if updates:
            task_updates[task] = updates

    if not task_updates:
        print("No adapters found!")
        return

    # Find common layers across all tasks
    all_layers = [set(updates.keys()) for updates in task_updates.values()]
    common_layers = sorted(set.intersection(*all_layers))
    print(f"\nFound {len(common_layers)} common layers across all tasks")

    # Define task categories based on the actual performance table
    category_groups = {
        'Analysis': ['formula', 'financebench', 'xbrl_extract'],
        'General': ['headline', 'ner', 'sentiment'],
        'Reporting': ['finer', 'xbrl_term'],
    }

    # Analyze attention layers (where most interference occurs)
    attention_layers = [l for l in common_layers if any(x in l for x in ['q_proj', 'k_proj', 'v_proj'])]
    print(f"Analyzing {len(attention_layers)} attention layers...")

    results = []

    print("\nComputing similarities...")

    # Within-category similarities
    for category, tasks in category_groups.items():
        if len(tasks) < 2:
            continue

        similarities = []
        for i, task1 in enumerate(tasks):
            for task2 in tasks:
                if task1 == task2:
                    continue
                print(task1, "<->", task2, end=": ")
                if task1 in task_updates and task2 in task_updates:
                    # Compute layer-wise similarities
                    layer_sims = []
                    for layer in attention_layers:
                        if layer in task_updates[task1] and layer in task_updates[task2]:
                            sim = cosine_similarity(
                                task_updates[task1][layer],
                                task_updates[task2][layer]
                            )
                            layer_sims.append(sim)

                    if layer_sims:
                        avg_sim = np.mean(layer_sims)
                        similarities.append(avg_sim)
                        print(f"{avg_sim:.4f}")

        if similarities:
            category_avg = np.mean(similarities)
            results.append({
                'Layer_Type': 'attention',
                'Category': category,
                'Relationship': 'Within-Category',
                'Avg_Similarity': category_avg,
                'Task_Pairs': len(similarities)
            })
            print(f"  → {category} within-category: {category_avg:.6f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('interference_evidence.csv', index=False)

    print(f"\n✓ Results saved to 'interference_evidence.csv'")
    print(f"✓ Generated {len(results)} similarity measurements")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    within_cat = df[df['Relationship'] == 'Within-Category']
    cross_cat = df[df['Relationship'] == 'Cross-Category']

    print("\nWithin-Category Similarities:")
    for _, row in within_cat.iterrows():
        print(f"  {row['Category']}: {row['Avg_Similarity']:.4f}")

    print("\nCross-Category Similarities:")
    for _, row in cross_cat.iterrows():
        print(f"  {row['Category']}: {row['Avg_Similarity']:.4f}")

    return df


if __name__ == "__main__":
    analyze_interference()