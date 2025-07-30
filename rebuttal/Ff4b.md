Dear NeurIPS Reviewer,

Thank you for the constructive suggestions.

## Q1 Multi-task vs Single-task Fine-tuning

### Experiment

We trained three new LoRA adapters (rank 8, 8-bit) by merging the training sets within each original task category and
running concurrent multi-task fine-tuning.

Before fine-tuning we balanced every category. When one dataset was much larger than the rest we performed random
under-sampling. When a dataset was much smaller we duplicated its samples to achieve random over-sampling. Final sample
counts will be provided in the supplementary materials of the camera-ready version.

We also introduced a three-shot prompting baseline.

| **Datasets**                           | Llama 3.1 8B | **Llama 3.1 8B (3 shot)** | Llama 3.1 8B LoRA 8bit-r8 (Single-task) | **Llama 3.1 8B LoRA 8bit-r8 (Multi-task)** |
|:---------------------------------------|:------------:|:-------------------------:|:---------------------------------------:|:------------------------------------------:|
| **General Financial Tasks**            |              |                           |                                         |                                            |
| FPB                                    |    68.73     |           76.40           |                **85.64**                |                   85.31                    |
| FiQA SA                                |    46.55     |           64.68           |                  81.28                  |                 **82.20**                  |
| TFNS                                   |    69.97     |           28.81           |                **88.02**                |                   34.51                    |
| NWGI                                   |    43.86     |           32.20           |                **54.16**                |                   36.51                    |
| NER                                    |    48.89     |           55.34           |                **98.05**                |                   76.07                    |
| Headline                               |    45.34     |           70.01           |                **84.66**                |                   13.90                    |
| **Financial Certification Tasks**      |              |                           |                                         |                                            |
| CFA 1                                  |    13.33     |           51.11           |                **86.67**                |                     -                      |
| CFA 2                                  |    19.48     |           37.66           |                **88.31**                |                     -                      |
| CFA 3                                  |    16.67     |           51.28           |                **70.51**                |                     -                      |
| CPA                                    |    31.68     |           45.54           |                **80.20**                |                     -                      |
| **Financial Reporting Tasks**          |              |                           |                                         |                                            |
| FiNER                                  |    21.28     |           30.76           |                **74.10**                |                    0.41                    |
| FNXL                                   |     3.64     |           13.15           |                **23.57**                |                    0.00                    |
| XBRL Term (BERTScore)                  |    0.574     |           0.595           |                  0.599                  |                 **0.676**                  |
| **Financial Statement Analysis Tasks** |              |                           |                                         |                                            |
| Tag Extraction                         |    69.16     |           70.22           |                **89.13**                |                   88.78                    |
| Value Extraction                       |    52.46     |           72.27           |                **98.49**                |                   97.62                    |
| Formula Construction                   |    12.92     |           17.73           |                  77.61                  |                 **83.33**                  |
| Formula Calculation                    |    27.27     |           33.65           |                  98.68                  |                 **99.04**                  |
| Finance Bench (BERTScore)              |    0.443     |           0.580           |                  0.511                  |                 **0.621**                  |
| Financial Math                         |    11.00     |           32.00           |                  30.00                  |                 **58.00**                  |
| **Overall Average**                    |              |                           |                                         |                                            |
| Aggregated                             |    37.05     |           47.38           |                **74.74**                |                   65.69                    |

_Financial Certification Tasks are shown as a single task because they share the same format in our original
submission._

*Due to character limit, the table only lists Accuracy. For BERTScore we report the F1 value.*

### Takeaways

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Tasks such as formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks appear to share
underlying knowledge like the structure of financial statements and basic numerical reasoning. Learning them together
helps the model build a broader and more useful representation.

In contrast, we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling the model struggles to optimize for very different objectives at the same time.

Overall performance drops when unrelated tasks are merged. Closely related tasks can benefit from joint training, while
divergent tasks often harm each other.

### Task similarity analysis

We will compute cosine similarity between TF-IDF vectors of instruction texts and present the resulting matrix in the
supplementary materials. This analysis will help confirm that tasks with higher textual similarity also support positive
transfer.

## Q2 Full fine-tuning vs LoRA baseline

A full-parameter fine-tuning baseline would indeed give an upper bound on performance and clarify the trade-off against
parameter efficient methods. Due to limited compute we could not finish full fine-tuning before the rebuttal deadline.
We have started a full fine-tuning run for XBRL Analysis and plan to include the results in the camera-ready version.

Our main contribution is a benchmark that compares several parameter efficient approaches such as LoRA, QLoRA, and DoRA
in a demanding financial context. These methods remain the only practical option for many financial institutions that
lack large compute clusters.

## Concluding answers

1. Could merging or restructuring tasks help:  
   Only when subtasks share both format and objective, as shown by our new experiments. We will publish guidelines for
   dataset merging on our documentation website and release per category adapters on HuggingFace so that future work can
   explore dynamic routing without retraining from scratch.

2. Need for a full fine-tuning baseline:   
   We could not provide results within the rebuttal period but have already begun full fine-tuning on XBRL Analysis.
   Results will appear in the camera-ready version.

We appreciate your feedback. Your comments led to a clearer view of inter-task transfer and a stronger benchmark.