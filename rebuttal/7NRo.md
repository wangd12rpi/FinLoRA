Dear NeurIPS Reviewer:

Thank you for the constructive suggestions.

## Q1 How does LoRA alleviate the issues highlighted in Table1?

| Error class                                              | Error in Table1                                | Why the base Llama-3-8B fails                                                                                                                        | How LoRA fixes it                                                                                                  |
|----------------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **1. Unfamiliarity with US GAAP tags and Hallucination** | Mis-tags \$2bn as `MajorityEquityInterest`     | `MajorityEquityInterest` is not part of the US GAAP taxonomy. The base model hallucinated and created an non-existing tag.                           | After seeing the US GAAP tags during the fine-tuning process, the model know the valid tag name.                   |
| **2. Unfamiliarity with financial formula**              | Picks 1,209,000,000 instead of 125,978,000,000 | Base model did not know the formula of equality multiplier. `EquityMultiplier = Assets / Equity`, therefore did not select the correct assets value. | Fine-tuning exposed the LLM with relevant financial concepts and formula, allowing it to select the correct value. |

Overall, LoRA succeeds because fine-tuning exposed the model with financial concepts.

## Q2 What are the most difficult points in financial data?

Financial data poses three main challenges:

1. Domain specific vocabulary: Many specialised terms appear rarely in general corpora, so the base LLM has limited
   exposure and often misinterprets them.

2. Mixed formats: A single report can contain narrative text, XML tags, tables, and raw numbers. The model must process
   information across these varied structures.

3. Long context windows: XBRL documents remain lengthy even after boilerplate is removed. Inputs often reach about four
   thousand tokens, and numbers required by the same formula may be thousands of tokens apart.

## Q3 Future research directions in fine-tuning: what drives adaptation performance?

### Single-task vs. multi-task fine-tuning

We performed additional experiments to compare single-task and multi-task LoRA fine-tuning and added an additional 3
shot baseline. The detailed results are summarized below:

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
| Aggregated                             |    41.52     |           47.38           |                **72.96**                |                   65.69                    |

_Financial Certification Tasks are shown as a single task because they share the same format in our original
submission._

*Due to character limit, the table only lists Accuracy. For BERTScore we report the F1 value.*

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Tasks such as formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks appear to share
underlying knowledge like the structure of financial statements and basic numerical reasoning. Learning them together
helps the model build a broader and more useful representation.

In contrast we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling the model struggles to optimize for very different objectives at the same time.

Overall performance drops when unrelated tasks are merged. Closely related tasks can benefit from joint training, while
divergent tasks often harm each other.

We plan to extend the multi-task experiment to other LoRA methods as well.

### Federated LoRA on complex tasks

We also evaluated Federated LoRA on the XBRL analysis tasks, a more complex scenario compared to our earlier
sentiment-task experiment. The preliminary results are shown below, and we will extend this to the XBRL tagging task
before the camera-ready submission

| Task             | Base      | Centralized LoRA | Federated LoRA |
|------------------|-----------|------------------|----------------|
| Tag-Extraction   | 69.16     | **89.13**        | 61.22          |
| Value-Extraction | 52.46     | **98.49**        | 61.51          |
| Formula-Constr.  | **19.92** | 77.61            | 15.48          |
| Formula-Calc.    | **27.27** | 98.68            | 22.62          |

*replace this text with results interpretation*

### Adapter capacity: lora rank vs task complexity, how did it affect performance?

We compared LoRA ranks of 4 and 8 and observed rank 8 provided higher accuracy across most tasks. However, certain
particularly challenging tasks, especially under multi-task fine-tuning, may benefit from even higher LoRA ranks (e.g.,
16 or 32). We plan to benchmark LoRA ranks above 8 to identify optimal configurations for complex scenarios.

#### Cross‑task interference & routing

Currently, adapters are selected manually according to task type. However we plan to adopt Mixture‑of‑LoRA‑Experts (
MoLE) routing which dynamically selects adapters via a small gating network. We are running a test on such method and
will append a brief result to supplementary materials for camera‑ready version.

We also fine-tuned a multi-task version with a total of only four LoRA adapters with merged training datasets from tasks
that are similar. This will mitigate some issues with too many adapters. The overall results are displayed below.
Overall we observed some knowledge transfer on some tasks and interference on other tasks.

## Q4 Additional Catastrophic Forgetting Experiments