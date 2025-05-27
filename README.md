# FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets

FinLoRA: Finetuning Quantized Llama3 and DeepSeek's V3/R1 Models into Financial Large Language Models Using Low-Rank Adaptation on GPUs

## Motivation

The proprietary BloombergGPT model, announced in April 2023, made the financial sector value the potentials of FinLLMs.
However, its train-from-scratch approach took one million GPU hours, which is expensive (around $3 million, at a price of $3 per
GPU hour in 2023).

Leveraging open-source models, e.g., Llama3 and DeepSeek's V3/R1 models, we adopt the LoRA fine-tuning method. The number of trainable parameters are
reduced to 0.01% of the full parameters, while the compute cost is less than $100.

## Financial Tasks

We want our models to have the ability to perform both general financial tasks and XBRL related tasks. We select the
following datasets to evaluate the models performance.

**About XBRL**: XBRL is a standard format for financial reporting. Regulators like the SEC requires public companies to file financial
statements using XBRL. XBRL is based on XML so it is complex and difficult to generate and interpret by humans. We are
interested in XBRL reporting and analysis.  
**XBRL Reporting**: Helping small and medium-business (SMBs) report in XBRL format.  
**XBRL Analysis**: Assisting the extraction and analysis of XBRL reports.
### General Financial Tasks

| Question Sets | Type                     | # Test Samples | Metrics      | Source                                                                          |   
|----------|--------------------------|----------------|--------------|---------------------------------------------------------------------------------|
| FPB      | Sentiment Analysis       | 970            | Accuracy, F1 | [HF](https://huggingface.co/datasets/TheFinAI/en-fpb)                           |
| FiQA SA  | Sentiment Analysis       | 234            | Accuracy, F1 | [HF](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification)    |
| TFNS     | Sentiment Analysis       | 2.4K           | Accuracy, F1 | [HF](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| NWGI     | Sentiment Analysis       | 4.1K           | Accuracy, F1 | [HF](https://huggingface.co/datasets/TheFinAI/NWGI_test)                        |
| Headline | Headline Analysis        | 20.5K          | Accuracy, F1 | [HF](https://huggingface.co/datasets/FinGPT/fingpt-headline-cls) |
| NER      | Named Entity Recognition | 3.5K           | Accuracy, F1 | [HF](https://huggingface.co/datasets/FinGPT/fingpt-ner-cls)                     |

### XBRL tasks


#### XBRL Reporting

| Question Sets      | Type    | # Test Samples | Metrics      | Source                                                         |
|---------------|---------|----------------|--------------|----------------------------------------------------------------|
| FiNER-139 [7] | Tagging | 100K           | Accuracy, F1 | [HF](https://huggingface.co/datasets/nlpaueb/finer-139?row=16) |
| FNXL [8]      | Tagging | 1K             | Accuracy, F1 | [GitHub](https://github.com/soummyaah/FNXL)                    |

#### XBRL Analysis

| Question Sets             | Type            | # Test Samples | Metrics  | Source                                                                                                                        |
|----------------------|-----------------|----------------|----------|-------------------------------------------------------------------------------------------------------------------------------|
| Financial Math [9]   | Math            | 1K             | Accuracy | [GitHub](https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/formulas_with_explanations_with_questions_with_gt.xlsx) |
| Tags Extraction      | XBRL Extraction | 150            | Accuracy | -                                                                                                                             |
| Values Extraction    | XBRL Extraction | 150            | Accuracy | -                                                                                                                             |
| Formula Calculations | XBRL Extraction | 150            | Accuracy | -                                                                                                                             |

## Training Datasets

We selected the following collection of training datasets.

| Datasets        | Type                     | # Train Samples | Source                                                                          |   
|-----------------|--------------------------|-----------------|---------------------------------------------------------------------------------|
| Sentiment       | Sentiment Analysis       | 76.8K           | [HF](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train)             |
| Headline        | Headline Analysis        | 82.2K           | [HF](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification)    |
| NER             | Named Entity Recognition | 13.5K           | [HF](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| FiNER-139       | XBRL Tagging             | 900K            | [HF](https://huggingface.co/datasets/TheFinAI/NWGI_test)                        |
| XBRL Extraction | XBRL Extraction          | -               | -                                                                               |

## File Structures

```
FinLoRA
├── test
│   ├── run_evaluate.sh
│   ├── run_test_all.sh
│   ├── test.py
│   ├── test_all.py
│   ├── fiqa.py
│   ├── fpb.py
│   ├── headline.py
│   ├── ner.py
│   ├── nwgi.py
│   ├── tfns.py
│   └── xbrl.py
├── data 
│   ├── gen_fin_data.ipynb
│   ├── xbrl_extract.ipynb
│   ├── process_multitask_data.py
│   ├── process_xbrl_data.py
│   ├── process_xbrl_formula.py
│   ├── process_xbrl_tag.py
│   ├── test
│   ├── train
├── environment.yml
└── src
    ├── LoRAMoE 
    ├── OpenFedLLM
    └── finetune
        ├── script_train.sh
        ├── train_lora.py
        └── utils.py
        
```

## Scenarios

### Cross-task Generalization (Mixture of LoRA Experts)

We started with single-task finetuning, i.e., finetune a LoRA adaptor for a task. We got good performance.

Mixture of LoRA Experts (LoRA-MoE): a LoRA module acts as an expert, a router network assigns weights, such as
in [X-LoRA](https://arxiv.org/pdf/2402.07148) [4]. X-LoRA is built on top of huggingface PEFT.

### Improving Performance and Scalability for Inference Stage

SLoRA [5] is designed for serving many LoRA adapters efficiently. It stores all adapters in the CPU memory and
fetches the adapters needed to GPU memory. We will deploy it on a cloud server.

Difficulty: Current SLoRA implementation does not work with HuggingFace, and does not support newer model like Llama 3.

### Distributed Training with Enhanced Privacy

Multiple institutions might want to collaborate to finetune a FinLLM using their private datasets. Using zero-Knowledge
Proofs (ZKPs) in
the finetuning stage allows enhanced data privacy.


[//]: # (Different user base, our model serve community, open-source well, we use finetuning)

[//]: # (assume large amount of user: )

[//]: # (e)

[//]: # (percentage)

[//]: # (compare results with icdcs)

## References

[1] Xiao-Yang Liu, Jie Zhang, Guoxuan Wang, Weiqing Tong, Anwar Walid. FinGPT-HPC: Efficient Pretraining and Finetuning
Large Language Models for Financial Applications with High-Performance Computing. IEEE ICDCS 2024.

[2] Mao, Y., Ge, Y., Fan, Y., Xu, W., Mi, Y., Hu, Z. and Gao, Y., 2024. A Survey on LoRA of Large Language Models. arXiv
preprint arXiv:2407.11046.

[3] Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, Weizhu Chen. A Note on LoRA, 2024. https://arxiv.org/abs/2404.05086

[4] E.L. Buehler, M.J. Buehler. X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language
Models with Applications in Protein Mechanics and Design}, https://arxiv.org/abs/2402.07148

[5] Sheng, Ying and Cao, Shiyi and Li. Dacheng and Hooper, et al. S-LoRA: Serving Thousands of Concurrent LoRA
Adapters, https://arxiv.org/pdf/2311.03285

[6] Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, Matt White, Meikang Qiu, Differentially Private
Low-Rank Adaptation of Large Language Model Using Federated Learning, https://arxiv.org/abs/2312.17493 ACM Transactions
on Management Information Systems, 2024.

[7] Loukas, L.; Fergadiotis, M.; Chalkidis, I.; Spyropoulou, E.; Malakasiotis, P.; Androutsopoulos, I.; and Paliouras,
G. 2022. FiNER: Financial Numeric Entity Recognition for XBRL Tagging. In Muresan, S.; Nakov, P.; and Villavicencio, A.,
eds., Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
Dublin, Ireland: Association for Computational Linguistics.

[8] Sharma, S.; Khatuya, S.; Hegde, M.; Shaikh, A.; Dasgupta, K.; Goyal, P.; and Ganguly, N. 2023. Financial Numeric
Extreme Labelling: A dataset and benchmarking. In Rogers,
A.; Boyd-Graber, J.; and Okazaki, N., eds., Findings of the Association for Computational Linguistics: ACL 2023,
3550–3561. Toronto, Canada.

[9] Han, S.; Kang, H.; Jin, B.; Xiao-Yang Liu; and Yang, S. Y. 2024. XBRL Agent: Leveraging Large Language Models for
Financial Report Analysis. In Proceedings of the 5th ACM
International Conference on AI in Finance, ICAIF ’24, 856–864. New York, NY, USA:

[10] Wang, K.; Patel, J.; Shen, C.; Kim, D.; Zhu, A.; Lin, A.; Borella, L.; Osborne, C.; White, M.; Yang, S.; and
Yanglet, K. X. Xiao-Yang Liu. 2024. A Report on Financial Regulations Challenge at COLING 2025. arXiv:2412.11159.
