Dear NeurIPS Reviewer:

Thank you for the constructive suggestions. 
## Data Imbalance 
We use single task fine-tuning for all experiments as of the version we submitted. Some tasks include multiple datasets that are in the same format and they are balanced by over-sampling small datasets. Therefore, the issues of imbalance within different tasks of the same category mentioned in limitation one does not posses any problem toward our experiment. 

## Alternative Baseline: few-shot method
We agree that low‑resource tasks merit few‑shot or in-context learning baselines. All 14 datasets have now been re‑evaluated with 3‑shot prompting using Llama‑3‑8B. The detailed results are in the table below. 

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

*Few shots results analysis here*

## Single-Task vs. Multi-Task Fine-tuning

We apologize for the lack of clarity about our fine-tuning method in the paper. The nine LoRA adapters resulted from single-task fine-tuning were mentioned in Table 2 in the Supplementary Material and published on Hugging Face. We will make sure to explicitly mention that in the camera-ready version. 

### Experiment
We trained three new LoRA adapters (rank 8, 8‑bit) by merging the training sets within each of our original task categories using concurrent multi-task fine-tuning. 

For balance, we rebalance each category before fine‑tuning.  We conduct random under-sampling of majority datasets when a category contains an extreme minority that are already large (e.g. Headline). Random oversampling were also used by duplication of minority datasets when a single dataset dominates.  Final counts will be presented in the supplementary materials in the camera ready version. 

### Take‑aways
We observed positive knowledge transfer in Financial Statement Analysis. Formula construction, calculation, Finance Bench, and financial math see improvement with the multi-task approach. This suggests these tasks are related enough and share similar underlying knowledge (e.g. understanding financial statements, performing calculations) that learning them together helps the model build a more generalized understanding, thereby improving overall performance on these tasks.

Conversely, negative interference is evident in the General Financial and Financial Reporting tasks. The multi-task model performs worse on tasks like TFNS, Headline, FiNER, and FNXL. We suspect the interference is due to task format and objective differences. The model likely encountered difficulty optimizing for all formats at once, even with balanced data sampling.

Overall we see a decrease in performance for multi-task. The main takeaway is not all tasks are suitable to be grouped together. Highly related tasks can benefit from being trained together. Divergent tasks (different formats or objectives) are prone to negative transfer when merged. 

### Task‑similarity analysis
We computed cosine similarity between TF‑IDF of instruction texts and we will display task similarity matrix in supplementary materials. This will help to validate our findings that similar financial tasks will facilitate knowledge transfer. 

