Dear NeurIPS Reviewer,

Thank you for the constructive suggestions.

## Q1 Multi-task vs Single-task Fine-tuning

### Experiment

We trained three new LoRA adapters (rank 8, 8-bit) by merging the training sets within each original task category and
running concurrent multi-task fine-tuning.

Before fine-tuning we balanced every category. For smaller datasets we duplicated its samples to achieve random
over-sampling. Final sample counts will be provided in the supplementary materials of the camera-ready version.

We also introduced a three-shot prompting baseline based on the suggestion from another reviewer.

| **Datasets**                                   | Llama 3.1 8B | **Llama 3.1 8B (3 shot)** | Llama 3.1 8B LoRA 8bit-r8 (Single-task) | **Llama 3.1 8B LoRA 8bit-r8 (Multi-task)** |
|:-----------------------------------------------|:------------:|:-------------------------:|:---------------------------------------:|:------------------------------------------:|
| **General Financial Tasks**                    |              |                           |                                         |                                            |
| FPB                                            |    68.73     |           76.40           |                **85.64**                |                   85.31                    |
| FiQA SA                                        |    46.55     |           64.68           |                  81.28                  |                 **82.20**                  |
| TFNS                                           |    69.97     |           28.81           |                **88.02**                |                   34.51                    |
| NWGI                                           |    43.86     |           32.20           |                **54.16**                |                   36.51                    |
| NER                                            |    48.89     |           55.34           |                **98.05**                |                   76.07                    |
| Headline                                       |    45.34     |           70.01           |                **84.66**                |                   13.90                    |
| **Financial Certification Tasks**              |              |                           |                                         |                                            |
| CFA 1                                          |    13.33     |           51.11           |                **86.67**                |                     -                      |
| CFA 2                                          |    19.48     |           37.66           |                **88.31**                |                     -                      |
| CFA 3                                          |    16.67     |           51.28           |                **70.51**                |                     -                      |
| CPA                                            |    31.68     |           45.54           |                **80.20**                |                     -                      |
| **Financial Reporting Tasks**                  |              |                           |                                         |                                            |
| FiNER                                          |    21.28     |           30.76           |                **74.10**                |                   67.97                    |
| FNXL                                           |     3.64     |           13.15           |                  23.57                  |                 **28.23**                  |
| XBRL Term (BERTScore)                          |    0.574     |           0.595           |                  0.599                  |                 **0.676**                  |
| **Financial Statement Analysis Tasks**         |              |                           |                                         |                                            |
| Tag Extraction                                 |    69.16     |           70.22           |                **89.13**                |                   88.78                    |
| Value Extraction                               |    52.46     |           72.27           |                **98.49**                |                   97.62                    |
| Formula Construction                           |    12.92     |           17.73           |                  77.61                  |                 **83.33**                  |
| Formula Calculation                            |    27.27     |           33.65           |                  98.68                  |                 **99.04**                  |
| Finance Bench (BERTScore)                      |    0.443     |           0.580           |                  0.511                  |                 **0.621**                  |
| Financial Math                                 |    11.00     |           32.00           |                  30.00                  |                 **58.00**                  |
| **Overall Average** (Using BERTScore F1 × 100) |              |                           |                                         |                                            |
| Aggregated                                     |    37.05     |           47.38           |                **74.74**                |                   68.78                    |

*Due to character limit, the table only lists Accuracy. For BERTScore we report the F1 value.*

_For financial certification tasks, there is no multi-task score as we consider them to be only one task due to similar
format. To compute overall average of multi-task, we take single task performance of financial certification task. _


### Takeaways

#### Single-task vs. Multi-task

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks share similar
underlying knowledge like the structure of financial statements and numerical reasoning. Learning them together
helps the model to achieve an enhanced understanding in these topics.

In contrast, we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling, the model struggles to optimize for very different objectives at the same time.

Overall performance drops by 10 points when tasks are merged. Closely related tasks can benefit from joint training,
while
divergent tasks often harm each other.

#### Three-shot prompting

On average, three-shot prompting raises the base model score by roughly 10 points. Gains are most visible
on certification and statement analysis tasks where the extra context helps the model recall domain knowledge.
Three-shot prompting never matches LoRA single-task fine-tuning. The LoRA fine-tuned adapters still clearly outperform
the three-shot baseline.

### Task similarity analysis

We will compute cosine similarity between instruction texts and present the resulting matrix in the
supplementary materials. This analysis will help confirm that tasks with higher textual similarity also support positive
transfer.

## Q2 Full-parameter fine-tuning vs LoRA baseline

A full-parameter fine-tuning baseline would give an upper bound on performance and clarify the trade-off against
parameter efficient methods. Due to limited computation resource we could not complete full fine-tuning.

Full fine-tuning the Llama 3.1 8B model requires updating all ~8 billion of its parameters. LoRA
setup (with rank 8) updates only ~4.7 million parameters - a reduction of over 1,700 times. We estimated full
fine-tuning demands a multi-GPU server with over 120 GB of VRAM and takes 8 GPU hours per epoch per task. We therefore
could not finish it within the rebuttal timeframe and we will add it as future work.

Our main contribution is a benchmark that compares several LoRA methods in the financial context. These methods remain
the only practical option for financial institutions that have limited computational resources.

---

We appreciate your feedback. Your comments led to a stronger benchmark.