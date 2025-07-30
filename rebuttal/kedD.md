Dear NeurIPS Reviewer:

Thank you for your constructive suggestions.

## Data Imbalance

In the submitted version we used single-task fine-tuning for every experiment. Some tasks contain several datasets that
share an identical format. We balanced those tasks by over-sampling the smaller datasets. As a result, the cross-dataset
imbalance mentioned in Limitation 1 does not affect our experiments.

## Alternative Baseline: few-shot method

We agree that low-resource tasks deserve few-shot or in-context learning baselines. We therefore re-evaluated all
fourteen datasets
with three-shot prompting using Llama-3-8B-Instruct. The detailed numbers appear in the table below.

| **Datasets**                           | **Llama 3.1 8B** Base | Ministral-8B Base | **Llama 3.1 8B LoRA 8bit-r8** (Single-task) | Ministral-8B LoRA 8bit-r8 (Single-task) |
|:---------------------------------------|:---------------------:|:-----------------:|:-------------------------------------------:|:---------------------------------------:|
| **General Financial Tasks**            |                       |                   |                                             |                                         |
| FPB                                    |         68.73         |       73.08       |                  **85.64**                  |                  86.71                  |
| FiQA SA                                |         46.55         |       52.86       |                    81.28                    |                  80.00                  |
| TFNS                                   |         69.97         |       22.07       |                  **88.02**                  |                  45.85                  |
| NWGI                                   |         43.86         |       21.25       |                  **54.16**                  |                  56.90                  |
| NER                                    |         48.89         |       58.61       |                  **98.05**                  |                  98.05                  |
| Headline                               |         45.34         |       62.64       |                  **84.66**                  |                  97.51                  |
| **Financial Certification Tasks**      |                       |                   |                                             |                                         |
| CFA 1                                  |         13.33         |       88.89       |                    86.67                    |                  87.77                  |
| CFA 2                                  |         19.48         |       94.80       |                    88.31                    |                  94.80                  |
| CFA 3                                  |         16.67         |       78.20       |                    70.51                    |                  78.20                  |
| CPA                                    |         31.68         |       87.12       |                    80.20                    |                  91.08                  |
| **Financial Reporting Tasks**          |                       |                   |                                             |                                         |
| FiNER                                  |         21.28         |       00.27       |                  **74.10**                  |                                         |
| FNXL                                   |         3.64          |       00.00       |                  **23.57**                  |                                         |
| XBRL Term (BERTScore)                  |         0.574         |       00.56       |                    0.599                    |                  0.672                  |
| **Financial Statement Analysis Tasks** |                       |                   |                                             |                                         |
| Tag Extraction                         |         69.16         |       74.15       |                  **89.13**                  |                  84.51                  |
| Value Extraction                       |         52.46         |       74.21       |                  **98.49**                  |                  98.80                  |
| Formula Construction                   |         12.92         |       11.91       |                    77.61                    |                  62.39                  |
| Formula Calculation                    |         27.27         |       47.62       |                    98.68                    |                  48.50                  |
| Finance Bench (BERTScore)              |         0.443         |       0.584       |                    0.511                    |                  0.617                  |
| Financial Math                         |         11.00         |       36.00       |                    30.00                    |                  0.464                  |
| **Overall Average**                    |                       |                   |                                             |                                         |
| Aggregated                             |         37.05         |       52.53       |                  **74.74**                  |                                         |


### Three-shot analysis

On average, three-shot prompting raises the base model score by 5.9 points. Gains are most visible
on certification and statement analysis tasks where the extra context helps the model recall domain knowledge.
Three-shot prompting never matches LoRA single-task fine-tuning. The multi-task adapter outperforms the three-shot
baseline
on 12 of the 14 datasets but still behind the single-task adapter because of negative transfer on several
general financial and reporting tasks.

## Single-Task vs. Multi-Task Fine-tuning

We apologize for the lack of clarity in our paper. The nine LoRA adapters obtained from single-task fine-tuning were
listed
in Table 2 of the Supplementary Material and are available on Hugging Face. We will state this explicitly in the
camera-ready version.

### Experiment

We trained three new LoRA adapters (rank 8, 8-bit) by merging the training sets within each original task category and
running concurrent multi-task fine-tuning.

Before fine-tuning we balanced every category. When one dataset was much larger than the rest we performed random
under-sampling. When a dataset was much smaller we duplicated its samples to achieve random over-sampling. Final sample
counts will be provided in the supplementary materials of the camera-ready version.

### Takeaways

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Tasks such as formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks appear to share
underlying knowledge like the structure of financial statements and basic numerical reasoning. Learning them together
helps the model build a broader and more useful representation.

In contrast we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling the model struggles to optimize for very different objectives at the same time.

Overall performance drops when unrelated tasks are merged. Closely related tasks can benefit from joint training, while
divergent tasks often harm each other.

### Task similarity analysis

We will compute cosine similarity between TF-IDF vectors of instruction texts and present the resulting matrix in the
supplementary materials. This analysis will help confirm that tasks with higher textual similarity also support positive
transfer.
