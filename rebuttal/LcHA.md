Dear NeurIPS Reviewer:

Thank you for the constructive suggestions. 
## Q1 Broader cross‑domain generalization

**New evaluation**. We have extended our “out‑of‑domain” suite beyond MMLU and GSM8K to three knowledge‑heavy benchmarks:

| Benchmark        | Metric | Zero‑shot (LLM‑3‑8B) | LoRA‑r8 single‑task |
| ---------------- | ------ | -------------------- | ------------------- |
| TriviaQA‑Open    | EM     | **\[…]**             | \[…]                |
| NaturalQuestions | F1     | **\[…]**             | \[…]                |
| CoQA             | F1     | **\[…]**             | \[…]                |

## Q2 Federated LoRA on complex tasks

We agree that running federated only on sentiment analysis might not be enough. We added an additional experiment on a four‑node federated learning setting on four XBRL Analysis subtasks. Totaling one epoch (same with non-federated version) and 40 rounds. 
**this one might need to be run again**

| Task             | Base      | Centralized LoRA | Federated LoRA |
| ---------------- | --------- | ---------------- | -------------- |
| Tag‑Extraction   | 69.16     | **89.13**        | 61.22          |
| Value‑Extraction | 52.46     | **98.49**        | 61.51          |
| Formula‑Constr.  | **19.92** | 77.61            | 15.48          |
| Formula‑Calc.    | **27.27** | 98.68            | 22.62          |
*results interpretation*

## Q3 Model‑family diversity

We have repeated the core LoRA experiment and base model testing on Ministral-8B-Instruct-2410. Zero‑shot baseline and LoRA 8-bit rank 8 scores on all 14 tasks will be added in Table 4. We aim to complete all 4 LoRA methods for Ministral 8B before camera-ready. 

| **Datasets**                           | **Llama 3.1 8B** Base | Ministral-8B Base | **Llama 3.1 8B LoRA 8bit-r8** (Single-task) | Ministral-8B LoRA 8bit-r8 (Single-task) |
| :------------------------------------- | :-------------------: | :---------------: | :-----------------------------------------: | :-------------------------------------: |
| **General Financial Tasks**            |                       |                   |                                             |                                         |
| FPB                                    |         68.73         |       73.08       |                  **85.64**                  |                                         |
| FiQA SA                                |         46.55         |       52.86       |                    81.28                    |                                         |
| TFNS                                   |         69.97         |       22.07       |                  **88.02**                  |                                         |
| NWGI                                   |         43.86         |       21.25       |                  **54.16**                  |                                         |
| NER                                    |         48.89         |       58.61       |                  **98.05**                  |                                         |
| Headline                               |         45.34         |       62.64       |                  **84.66**                  |                                         |
| **Financial Certification Tasks**      |                       |                   |                                             |                                         |
| CFA 1                                  |                       |                   |                                             |                                         |
| CFA 2                                  |                       |                   |                                             |                                         |
| CFA 3                                  |                       |                   |                                             |                                         |
| CPA                                    |                       |                   |                                             |                                         |
| **Financial Reporting Tasks**          |                       |                   |                                             |                                         |
| FiNER                                  |         21.28         |                   |                  **74.10**                  |                                         |
| FNXL                                   |         3.64          |       0.00        |                  **23.57**                  |                                         |
| XBRL Term (BERTScore)                  |         0.574         |       0.563       |                    0.599                    |                                         |
| **Financial Statement Analysis Tasks** |                       |                   |                                             |                                         |
| Tag Extraction                         |         69.16         |       74.15       |                  **89.13**                  |                                         |
| Value Extraction                       |         52.46         |       74.21       |                  **98.49**                  |                                         |
| Formula Construction                   |         12.92         |       11.91       |                    77.61                    |                                         |
| Formula Calculation                    |         27.27         |       47.62       |                    98.68                    |                                         |
| Finance Bench (BERTScore)              |         0.443         |       0.584       |                    0.511                    |                                         |
| Financial Math                         |         11.00         |       36.00       |                    30.00                    |                                         |
| **Overall Average**                    |                       |                   |                                             |                                         |
| Aggregated                             |         41.52         |                   |                  **72.96**                  |                                         |


## Q4 Task difficulty calibration & answer validation

- CFA I/II/III, CPA‑REG. All are multiple‑choice questions sourced from publicly available mock exams; answer keys are provided by the publishers. Difficulty should be similar to the actual exam. Sample inputs/outputs are available in supplementary materials Section A.3.1. 
- XBRL Analysis. Questions are auto‑generated from 150 10‑K filings using five natural‑language templates. Ground‑truth answers are extracted with Arelle library, a widely used open‑source XBRL tool. 

## Q5 Multiple adapters at inference

Currently, adapters are selected manually according to task type. However we plan to adopt Mixture‑of‑LoRA‑Experts (MoLE) routing which dynamically selects adapters via a small gating network. We are running a test on such method and will append a brief result to supplementary materials for camera‑ready version. 

We also fine-tuned a multi-task version with a total of only four LoRA adapters with merged training datasets from tasks that are similar. This will mitigate some issues with too many adapters. The overall results are displayed below. Overall we observed some knowledge transfer on some tasks and interference on other tasks. 

## Q6 Domain shift at larger ranks / longer training

TODO

## Q7 Prompt format standardization

All datasets uses standardized prompt format. We will publish the instruction template in supplementary materials before camera-ready. For most tasks we follow the instruction format from prior work such as FinBen [1] and PIXIU [2]. For XBRL Analysis tasks here is the basic format:

```
<|system|>
You are a knowledgeable XBRL assistant. Your task is to ... 
<|user|>
Example Question: What is ... 
Example Answer: 500000
XML File: [XBRL Text]
Question: What is ...
Answer: 
<|assistant|>
```

## Q8 Bias, fairness, and compliance risks

We agree that financial LLMs require additional safeguards. While an in‑depth audit is beyond the scope of this benchmark, we have started a follow‑up project in collaboration with FINOS (Linux Foundation) AI governance framework to derive fairness test cases from financial statement analysis. 

**XBRL‑specific ethics.** XBRL are highly standardized, each report must be valid XML and follow the SEC guildlines on selecting US GAAP tags. However, we note potential coverage bias—e.g., Dow Jones 30 firms over‑represent certain sectors. We will add a remark to the dataset description that the dataset should not be used to benchmark sector‑level fairness.

[1]
[2]