# Summary of Rebuttal Experiments and Key Findings

We sincerely thank all the reviewers for their constructive and detailed feedback. To address the points raised, we
conducted several new experiments and analyses which have strengthened our paper. Below is a summary of
the key additions and findings.

## Expanded Model Diversity

To test the generalizability of our findings across different models, we evaluated and fine-tuned a new open-source base
model, **Ministral-8B**. The results showed that while the base Ministral-8B model had different starting performance
characteristics than Llama 3.1 8B, LoRA fine-tuning provided similarly significant gains. The final performance of both
the fine-tuned Llama and Ministral models was comparable, demonstrating that the benefits of LoRA are not limited to a
single model family.


## Multi-task vs. Single-task Fine-tuning and New Baselines

In response to reviewer suggestions, we conducted extensive new experiments comparing single-task and multi-task
fine-tuning and introduced new baselines to strengthen our evaluation.

* **Multi-task Fine-tuning**: We fine-tuned new LoRA adapters by merging datasets within our original task categories.
  We found **mixed results**:
    * **Positive Transfer**: Tasks in the **"Financial Statement Analysis"** category, which share underlying knowledge
      of financial formulas and numerical reasoning, showed clear performance gains from multi-task training.
    * **Negative Transfer**: Dissimilar tasks in the **"General Financial"** and **"Financial Reporting"** categories
      experienced performance drops due to conflicting task formats and objectives.

* **New Baselines**:
    * We added a 3-shot in-context learning baseline for Llama 3.1 8B. It improved performance over zero-shot by ~10
      points on average but was consistently outperformed by single-task LoRA fine-tuning.

## Task Similarity Analysis for Improved Multi-task Grouping

To better understand the mixed multi-task results, we conducted a task similarity analysis to provide a principled
method for grouping tasks.

* **Key Finding**: We used sentence embeddings to compute the cosine similarity between all 19 datasets and found a 
  strong correlation between task similarity and multi-task performance. Categories with high average similarity (
  e.g., Financial Statement Analysis, avg. similarity: 0.565) benefited from joint training, while those with low
  similarity (e.g., General Financial, avg. similarity: 0.345) suffered from negative transfer. We confirmed this by
  analyzing the cosine similarity of the LoRA adapter weights themselves.

* **New Task Clustering**: Based on a full similarity matrix, we proposed a new, more cohesive task clustering (
  e.g., "News & Sentiment," "Financial Statement Reasoning"). These clusters have significantly higher intra-cluster
  similarity and are expected to yield better results in future multi-task fine-tuning, which we will explore for the
  camera-ready version.

## Expanded Evaluations on Federated Learning and Catastrophic Forgetting

We expanded our evaluations to address reviewer concerns about the scope of our experiments on more advanced and
practical scenarios.

* **Federated LoRA on Complex Tasks**: We extended our federated learning evaluation beyond simple sentiment tasks to
  the more complex **XBRL Analysis tasks**. The results showed that Federated LoRA improved performance over the base
  model but did not reach the level of centralized LoRA, highlighting the challenges of federated training on complex,
  long-context tasks.

* **Catastrophic Forgetting**: We broadened our out-of-domain evaluation by testing on **TriviaQA-Open and CoQA** in
  addition to MMLU and GSM8K, and evaluated them on higher rank (32) LoRA models as well. The results confirmed our
  initial findings, showing minimal catastrophic forgetting of general knowledge after domain-specific LoRA
  fine-tuning.

---

We believe these new experiments and analyses have robustly addressed the reviewers' concerns and improved
the contribution of our work. Thank you again for your valuable guidance.

