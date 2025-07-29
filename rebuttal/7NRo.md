## Q1 How does LoRA alleviate the issues highlighted in Table1?

| Error class                                              | Error in Table1                                | Why the base Llama‑3‑8B fails                                                                                                                        | How LoRA fixes it                                                                                                  |
|----------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **1. Unfamiliarity with US GAAP tags and Hallucination** | Mis‑tags \$2bn as `MajorityEquityInterest`     | `MajorityEquityInterest` is not part of the US GAAP taxonomy. The base model hallucinated and created an non-existing tag.                           | After seeing the US GAAP tags during the fine-tuning process, the model know the valid tag name.                   |
| **2. Unfamiliarity with financial formula**              | Picks 1,209,000,000 instead of 125,978,000,000 | Base model did not know the formula of equality multiplier. `EquityMultiplier = Assets / Equity`, therefore did not select the correct assets value. | Fine-tuning exposed the LLM with relevant financial concepts and formula, allowing it to select the correct value. |

Overall, LoRA succeeds because fine-tuning exposed the model with financial concepts.

## Q2 What are the most difficult points in financial data?

1. highly professional: a lot of the concepts and words are not represented enough when the base llm is trained becaause
   its rarely occured in common datasets.
2. Mix of formats. finacial data includes differnt text formats, like descriptive text, XML, tables, and raw numbers all
   appear in the same document.
3. Long context. for xbrl the document is very long. even when broken down and stripped of unnassary information the
   input length can still be around 4k tokens. Important numbers that are a part of a formula may be thousands of tokens
   apart.

## Q3 Future research directions in fine‑tuning - what drives adaptation performance?

#### Single-task vs. multi-task fine-tuning

We performed additional experiments to compare single-task and multi-task LoRA fine-tuning. The detailed results are
summarized below:

| **Datasets**                           | **Llama 3.1 8B** | **Llama 3.1 8B (3 shot)** | **Llama 3.1 8B LoRA 8bit-r8** (Single-task) | **Llama 3.1 8B LoRA 8bit-r8 (Multi-task)** |
|:---------------------------------------|:----------------:|:-------------------------:|:-------------------------------------------:|:------------------------------------------:|
| **General Financial Tasks**            |                  |                           |                                             |                                            |
| FPB                                    |      68.73       |             -             |                  **85.64**                  |                   85.31                    |
| FiQA SA                                |      46.55       |             -             |                    81.28                    |                 **82.20**                  |
| TFNS                                   |      69.97       |             -             |                  **88.02**                  |                   34.51                    |
| NWGI                                   |      43.86       |             -             |                  **54.16**                  |                   36.51                    |
| NER                                    |      48.89       |             -             |                  **98.05**                  |                   76.07                    |
| Headline                               |      45.34       |             -             |                  **84.66**                  |                   13.90                    |
| **Financial Reporting Tasks**          |                  |                           |                                             |                                            |
| FiNER                                  |      21.28       |             -             |                  **74.10**                  |                    0.41                    |
| FNXL                                   |       3.64       |             -             |                  **23.57**                  |                    0.00                    |
| XBRL Term (BERTScore)                  |      0.574       |             -             |                    0.599                    |                 **0.676**                  |
| **Financial Statement Analysis Tasks** |                  |                           |                                             |                                            |
| Tag Extraction                         |      69.16       |             -             |                  **89.13**                  |                   88.78                    |
| Value Extraction                       |      52.46       |             -             |                  **98.49**                  |                   97.62                    |
| Formula Construction                   |      12.92       |             -             |                    77.61                    |                 **83.33**                  |
| Formula Calculation                    |      27.27       |             -             |                    98.68                    |                 **99.04**                  |
| Finance Bench (BERTScore)              |      0.443       |             -             |                    0.511                    |                 **0.621**                  |
| Financial Math                         |      11.00       |             -             |                    30.00                    |                 **58.00**                  |
| **Overall Average**                    |                  |                           |                                             |                                            |
| Aggregated                             |      41.52       |             -             |                  **72.96**                  |                   65.69                    |

_Financial Certification Tasks are omitted in this table because we already trained them them together in our original
submission_
*The table only includes Accuracy (F1 for BERTScore) due to character limit.*

We observed positive knowledge transfer in Financial Statement Analysis. Formula construction, calculation, Finance
Bench, and financial math see improvement with the multi-task approach. This suggests these tasks are related enough and
share similar underlying knowledge (e.g. understanding financial statements, performing calculations) that learning them
together helps the model build a more generalized understanding, thereby improving overall performance on these tasks.

Conversely, negative interference is evident in the General Financial and Financial Reporting tasks. The multi-task
model performs worse on tasks like TFNS, Headline, FiNER, and FNXL. We suspect the interference is due to task format
and objective differences. The model likely encountered difficulty optimizing for all formats at once, even with
balanced data sampling.

We plan to extend the multi-task experiment to other LoRA methods as well.

#### Federated LoRA on complex tasks

We also evaluated Federated LoRA on the XBRL analysis tasks, a more complex scenario compared to our earlier
sentiment-task experiment. The preliminary results are shown below, and we will extend this to the XBRL tagging task
before the camera-ready submission

| Task             | Base      | Centralized LoRA | Federated LoRA |
|------------------|-----------|------------------|----------------|
| Tag‑Extraction   | 69.16     | **89.13**        | 61.22          |
| Value‑Extraction | 52.46     | **98.49**        | 61.51          |
| Formula‑Constr.  | **19.92** | 77.61            | 15.48          |
| Formula‑Calc.    | **27.27** | 98.68            | 22.62          |

*results interpretation*

#### Adapter capacity: lora rank vs task complexity, how did it affecs performance?

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

