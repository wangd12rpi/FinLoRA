Dear NeurIPS Reviewer,

Thank you for the constructive suggestions.

## Q1 Broader Catastrophic Forgetting Study

We have extended our “out-of-domain” suite beyond MMLU and GSM8K to three additional knowledge-heavy benchmarks:

| Benchmark        | Metric | Zero-shot (LLM-3-8B) | LoRA-r8 single-task |
|------------------|--------|----------------------|---------------------|
| TriviaQA-Open    | EM     |                      |                     |
| NaturalQuestions | F1     |                      |                     |
| CoQA             | F1     |                      |                     |

## Q2 Federated LoRA on Complex Tasks

Running federated learning only on sentiment analysis was not sufficient. We therefore added an experiment with a
four-node federated setting covering four XBRL Analysis subtasks. The experiment uses one epoch (identical to the
non-federated version) and 30 rounds.

| Task             | Base  | Centralized LoRA | Federated LoRA |
|:-----------------|:------|:-----------------|:---------------|
| Tag-Extraction   | 69.16 | **89.13**        | 69.04          |
| Value-Extraction | 52.46 | **98.49**        | 79.76          |
| Formula-Constr.  | 19.92 | **77.61**        | 13.10          |
| Formula-Calc.    | 27.27 | **98.68**        | 28.57          |
| **Average**      | 42.20 | **90.98**        | 47.62          |

Federated training noticeably improves value extraction, matches tag extraction and formula calculation, and slightly
reduces formula construction. Overall, performance improves relative to the base model, although centralized LoRA still
performs best.

## Q3 Model Diversity

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

The Ministral-8B base model outperforms the Llama 3.1 base on several financial tasks (e.g., FPB,
Headline) but underperforms on TFNS and some reporting benchmarks. Overall Ministral-8b performs better than Llama 3.1
8B base. After single-task
LoRA fine-tuning, the Ministral variant gains sizeable improvements on most tasks—often close to the tuned
Llama model and in a few cases surpassing it (e.g., FPB, Value Extraction). However, performance remains uneven on tasks
where the base was already weak (TFNS, Formula Calculation). 

## Q4 Task Difficulty & Ground-Truth Validity

* CFA I/II/III and CPA-REG are multiple-choice questions sourced from publicly available mock exams; answer keys are
  provided by the publishers. Difficulty is comparable to the actual exams. Sample inputs and outputs are available in
  Supplementary Materials Section A.3.1.
* XBRL Analysis questions are auto-generated from 150 10-K filings using five natural-language templates. Ground-truth
  answers are extracted with the Arelle library, a widely used open-source XBRL tool.

## Q5 Multiple Adapters at Inference

Adapters are currently selected manually according to task type. We plan to adopt a Mixture-of-LoRA-Experts (MoLE)
router that selects adapters with a small gating network. Preliminary results will be added to the supplementary
materials for the camera-ready version.

We also fine-tuned a multitask variant that uses only four LoRA adapters trained on merged datasets from related tasks.
This reduces the total number of adapters. We observe knowledge transfer on some tasks and negative interference on
others.

## Q6 Domain Shift at Larger Ranks and Longer Training

To further measure the effect of larger ranks and extended training, we will conduct new experiment that increase the
LoRA rank from 8 to 32 and trained on two representative financial tasks (Sentiment Analysis and XBRL Analysis) for 8
epoch and measure the effects
on general knowledge datasets. We will present the results for the camera-ready version.

## Q7 Prompt Formatting

All datasets use a standardized prompt format. We will publish the instruction template in the supplementary materials
before the camera-ready deadline. For most tasks we follow the instruction formats from prior work such as FinBen [1]
and PIXIU [2]. The basic template for XBRL Analysis tasks is:

```
<|system|>
You are a knowledgeable XBRL assistant. Your task is to ... 
<|user|>
Example Question: What is ... Example Answer: 500000 
XML File: [XBRL Text] 
Question: What is ... Answer: 
<|assistant|>
```

## Q8 Bias, Fairness, and Compliance

We agree that financial LLMs require additional safeguards. While an in-depth audit is beyond the scope of this
benchmark, we have started a follow-up project in collaboration with the FINOS (Linux Foundation) AI governance
framework to derive fairness test cases from financial statement analysis.

**XBRL‑specific ethics.** XBRL are highly standardized, each report must be valid XML and follow the SEC guildlines on
selecting US GAAP tags. However, we note potential coverage bias—e.g., Dow Jones 30 firms over‑represent certain
sectors. We will add a remark to the dataset description that the dataset should not be used to benchmark sector‑level
fairness.