Dear NeurIPS Reviewer:

Thank you for the constructive suggestions. 
## Multi-task vs Single-task Fine-tuning
### Experiment 
We trained three new LoRA adapters (rank 8, 8‑bit) by merging the training sets within each of our original task categories using concurrent multi-task fine-tuning. 

For balance, we rebalance each category before fine‑tuning.  We conduct random under-sampling of majority datasets when a category contains an extreme minority that are already large (e.g. Headline). Random oversampling were also used by duplication of minority datasets when a single dataset dominates.  Final counts will be presented in the supplementary materials in the camera ready version. 

We also added 3 shot as one our baseline based on the suggestions of another reviewer. 

| **Datasets**                           | **Llama 3.1 8B** | **Llama 3.1 8B (3 shot)** | **Llama 3.1 8B LoRA 8bit-r8** (Single-task) | **Llama 3.1 8B LoRA 8bit-r8 (Multi-task)** |
| :------------------------------------- | :--------------: | :-----------------------: | :-----------------------------------------: | :----------------------------------------: |
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
_Financial Certification Tasks are omitted in this table because we already trained them them together in our original submission_
*The table only includes Accuracy (F1 for BERTScore) due to character limit.* 

### Take‑aways
We observed positive knowledge transfer in Financial Statement Analysis. Formula construction/calculation, Finance Bench, and financial math see improvement with the multi-task approach. This suggests these tasks are related enough and share similar underlying knowledge (e.g. understanding financial statements, performing calculations) that learning them together helps the model build a more generalized understanding, thereby improving performance.

Conversely, negative interference is evident in the General Financial and Financial Reporting tasks. The multi-task model performs worse on tasks like TFNS, Headline, FiNER, and FNXL. We suspect the interference is due to task format and objective differences. The model likely encountered difficulty optimizing for all formats at once, even with balanced data sampling.

Overall we see a decrease in performance for multi-task. The main takeaway is not all tasks are suitable to be grouped together. Highly related tasks can benefit from being trained together. Divergent tasks (different formats or objectives) are prone to negative transfer when merged. 

### Task‑similarity analysis
We computed cosine similarity between TF‑IDF of instruction texts and we will display task similarity matrix in supplementary materials. This will help to validate our findings that similar financial tasks will facilitate knowledge transfer. 

## Full Finetuning vs LoRA baseline
We agree with that a full-parameter fine-tuning baseline would be a valuable addition, providing an upper bound on performance and a clearer trade-off analysis against PEFT methods. Unfortunately, due to significant computational and time constraints, we were unable to complete full fine-tuning experiments for the rebuttal deadline. However we have launched a full fine-tuning experiment for XBRL Analysis and we would expect results to be added for camera-ready version. 

The primary contribution of our paper is to establish a benchmark for comparing the performance of various PEFT methods (e.g., LoRA, QLoRA, DoRA) in the complex financial domain. These methods are often the only practical option for financial organizations without access to massive computational resources. 

## Concluding answers

1. Could merging or restructuring tasks be beneficial? Only for subtasks with similar format and objective, as our new experiments show; otherwise interference dominates. We will add guidelines on dataset merging on our documentation website and release per‑category adapters on HuggingFace so future work can explore dynamic routing without re‑training from scratch.
2. Need for full fine‑tuning baseline? Due to time constraint we are unable to provide any results at this time. However we have started full fine-tuning on XBRL Analysis and results will be added to the camera ready version. 

We appreciate your feedback. It led to a clearer picture of inter‑task transfer and a stronger benchmark.