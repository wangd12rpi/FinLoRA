Dear NeurIPS Reviewer:

Thank you for your constructive suggestions.

## Q1 Data Imbalance

We appreciate that you are highlighting the important issue of data imbalance across tasks. We would like to clarify a
potential misunderstanding. In the submitted version, all experiments were
conducted using a single-task fine-tuning approach. Consequently, the imbalance between tasks within a major category
did not affect the results. Some tasks contain several datasets that share an identical format. We balanced those tasks
by over-sampling the smaller
datasets. As a result, the cross-dataset imbalance mentioned in Limitation 1 does not affect our experiments.

However, your point is still highly relevant and crucial for multi-task learning.
Inspired by this feedback, we have added a new suite of multi-task fine-tuning experiments in our revision (see Section
Q3). In this new setup, we directly address the imbalance issue by implementing a rebalancing strategy that over-samples
data from smaller datasets.

## Q2 Alternative Baseline: few-shot method

We agree that low-resource tasks deserve few-shot or in-context learning baselines. We therefore re-evaluated all
19 datasets with three-shot prompting using Llama-3-8B-Instruct. The detailed numbers appear in the table below.

[see table on ff4b]

*Due to character limit, the table only lists Accuracy. For BERTScore we report the F1 value.*

_For financial certification tasks, there is no multi-task score as we consider them to be only one task due to similar
format_

### Experiment

For each test question, we randomly select three question answer pairs from the training set and use them for three shot
prompting. The format is similar to the following: 
```
Instruction: what is the sentiment...
Input: The prices...
Answer: negative
Input: Sales from...
Answer: positive
Input: The company...
Answer: positive
Input: The stock...
Answer: 
```

### Analysis

On average, three-shot prompting raises the base model score by roughly 10 points. Gains are most visible
on certification and statement analysis tasks where the extra context helps the model recall domain knowledge.
Three-shot prompting never matches LoRA single-task fine-tuning. The LoRA fine-tuned adapters still clearly outperform
the three-shot baseline.

## Q3 Single-Task vs. Multi-Task Fine-tuning

We apologize for the lack of clarity in our paper. The original nine LoRA adapters obtained from single-task
fine-tuning. They were listed in Table 2 of the Supplementary Material and are available on Hugging Face. We will state
this explicitly in the camera-ready version.

### Experiment

We trained three new LoRA adapters (rank 8, 8-bit) by merging the training sets within each original task category and
running concurrent multi-task fine-tuning.

Before fine-tuning we balanced every category. For smaller datasets we duplicated its samples to achieve random
over-sampling. Final sample counts will be provided in the supplementary materials of the camera-ready version.

### Takeaways

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks share similar
underlying knowledge like the structure of financial statements and numerical reasoning. Learning them together
helps the model to achieve an enhanced understanding in these topics.

In contrast, we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling, the model struggles to optimize for very different objectives at the same time.

Overall, performance drops by 11 points when tasks are merged. Closely related tasks can benefit from joint training, while
divergent tasks often harm each other.

### Task similarity analysis

We will compute cosine similarity between instruction texts and present the resulting matrix in the
supplementary materials. This analysis will help confirm that tasks with higher textual similarity also support positive
transfer.

---

We appreciate your feedback. Your comments led to a stronger benchmark.
