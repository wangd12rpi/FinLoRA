Dear reviewer:

Thank you for your further clarification of the question. Your idea is very inspiring, and we agree statistics like task
similarity can help us
mitigate interference.

## 1. Task similarity

To study which tasks are similar or different, we used a sentence embedding method. For each dataset, we randomly
sampled up to 200 training examples. We concatenate question and answer and use the `all-mpnet-base-v2`
sentence-transformer model to turn them into embeddings.
We took the average embedding per dataset and computed the cosine similarity between all pairs. This gives a
value from 0 to 1: higher means the tasks are semantically more similar.

### Similarity results

#### General Category

|                    | Sentiment Analysis | Headline |   NER |
|:-------------------|-------------------:|---------:|------:|
| Sentiment Analysis |              1.000 |    0.556 | 0.244 |
| Headline           |              0.556 |    1.000 | 0.236 |
| NER                |              0.244 |    0.236 | 1.000 |

#### Reporting Category

|           | XBRL‑Term | FiNER |
|:----------|----------:|------:|
| XBRL‑Term |     1.000 | 0.489 |
| FiNER     |     0.489 | 1.000 |

#### Analysis Category

|                | Tag‑Extraction | Financial Math | FinanceBench |
|:---------------|---------------:|---------------:|-------------:|
| Tag‑Extraction |          1.000 |          0.472 |        0.579 |
| Financial Math |          0.472 |          1.000 |        0.644 |
| FinanceBench   |          0.579 |          0.644 |        1.000 |

### Result Interpretation

These averages line up with what we saw in the multi-task LoRA evaluation, showing that it can be a reliable way to
indicate what tasks/datasets are better to be fine-tuned together.

* **General (avg. similarity 0.345): Mixed or negative**  
  Low similarity warned us about conflict. Performance of TFNS and Headline, which rely on different input styles,
  reduced significantly when trained together. NER also fell −22 pts. Only FiQA SA improved slightly.

* **Reporting (avg. similarity 0.489): Mild gains**  
  Similarity is moderate. Multi-task helped FiNER (+4 pts) and XBRL-Term (+8 pts, BERTScore), but the improvements were
  smaller than in Analysis. (Reporting results were update, please refer to section 3)

* **Analysis (avg. similarity 0.565): Mostly gains**  
  Higher similarity results in knowledge transfer. Financial Math jumped +28 pts, Formula Construction +6 pts, and
  FinanceBench +11 pts under multi-task tuning.

### Improved Tasks Clustering for Multi-task Fine-tuning

We have computed the complete 19x19 similarity matrix across all datasets (not shown here to save space). By clustering
using the matrix, we can form new, tighter groups for multi-task fine-tuning.

| Cluster                          | Tasks (members)                                                                                                                                   | Mean Similarity |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|
| 1. News & Sentiment              | FiQA, FPB, NWGI, TFNS, Headline                                                                                                                   |       **0.769** |
| 2. NER & Certification           | CFA‑1, CFA‑2, CFA‑3, CPA-REG, NER                                                                                                                 |       **0.580** |
| 3. Financial Statement Reasoning | FinanceBench, FiNER, FNXL, FinancialMath, XBRL‑Formula-Calculation, XBRL‑Formula-Construction, XBRL‑Tag-Extraction, XBRL‑Val-Extration, XBRL‑Term |       **0.613** |

The mean similarity is higher compared to our original categories, and we expect less interference with multi-task
fine-tuning compared to our original grouping. We plan to conduct new multi-task fine-tuning based on
new grouping and display evaluation results before camera-ready.

## 2. Strengths of Financial Benchmark Compared to Other Benchmarks

Comparing to other financial benchmarks like FinBen [1], our benchmark

- Includes full training and test sets, not just evaluation data, and provides all trained LoRA adapters for
  reproducible fine-tuning.
- Structured for both single- and multi-task learning and interference analysis.
- Covers more professional-level tasks (e.g., XBRL tasks, certification exams), and introduces novel XBRL Analysis
  tasks.

## 3. Update on Previous Multi-task Result

In our previous rebuttal response, an error was made on the multi-task experiment results for the financial reporting
tasks. Under the tight time constraints during rebuttal, we incorrectly set the delimiter that we use to
parse the output for the FiNER and FNXL evaluation, leading to the low scores. We apologize for this mistake in
execution and present the corrected results and updated average below:

| **Datasets**                | Llama 3.1 8B (Base) | **Llama 3.1 8B (3-shot)** | Llama 3.1 8B LoRA (Single-task) | **Llama 3.1 8B LoRA (Multi-task)** |
|:----------------------------|:-------------------:|:-------------------------:|:-------------------------------:|:----------------------------------:|
| FiNER                       |        21.28        |           30.76           |            **74.10**            |               67.97                |
| FNXL                        |        03.64        |           13.15           |              23.57              |             **28.23**              |
| XBRL Term (BERTScore)       |        0.574        |           0.595           |              0.599              |             **0.676**              |
| **Updated Overall Average** |        37.05        |           47.38           |            **74.74**            |               68.78                |

We apologize for the earlier error, and thank you for the opportunity to correct it.

---

Sincerely,

The Authors


---
[1] Qianqian Xie et al. FinBen: An holistic financial benchmark for large language models. In The
Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks
Track, 2024.


---
Dear reviewer:

Thank you for your response.

## Key challenges

- **Long & uneven context.** XBRL snippets and filings are long and uneven in length, stressing context windows and GPU
  memory. **Addressed:** we de‑noise inputs, remove redundant chars, and drop Q/A pairs exceeding the window (for
  training only); we also use length‑aware batching and gradient accumulation.

- **Numeric reasoning & normalization.** Multi‑step math, unit scaling (millions vs thousands), parentheses, and many
  valid numerical forms (“1.5B” or “$1,500,000,000”) complicate LLM inference and evaluation scoring. **Addressed:**
  unit‑aware normalization and more tolerant evaluation (rounding, accepting alternative math formula writing style).

- **Class imbalance.** Rare labels (e.g., in FiNER, FNXL) or tasks can be under‑learned without weighting, specifically
  for XBRL tagging tasks (FiNER, FNXL). . **Addressed:** we use oversampling and data augmentation for minority classes.

- **Training instability.** Mixed formats (e.g., different prompt style from different datasets for the same) and length
  variance can destabilize the fine-tuning process. **Addressed:** extensive hyperparameter tuning to stabilize the
  fine-tuning process, early stopping to avoid overfitting, and learning rate warmup.

---

As we approach the end of the discussion period, we want to again express our sincere thanks for your thorough engagement. Your feedback has been instrumental in improving our paper, and we have worked carefully to address all of your concerns.

We hope our current responses have been satisfactory, and if so, we would be grateful if you would consider this in your updated score. Please know that we welcome any further questions, and we are eager not only to answer them but also to incorporate any further guidance to strengthen our work.

Thank you again for your time and valuable insights.

Sincerely,

The Authors
