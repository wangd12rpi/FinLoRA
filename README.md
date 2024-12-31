# FinLoRA: Finetuning Quantized Financial Large Language Models Using Low-Rank Adaptation

### Motivation

The closed-source BloombergGPT was announced in April 2023, then the financial sector started to value FinLLMs. However,
its train-from-scratch approach requires millions of GPU hours, which is too expensive. Instead, we adopt the LoRA
fine-tuning approach to leverage open-source models like Llama. The trainable parameters are reduced to only 0.01% of
the full parameters. The trainable parameters are reduced to as low as only 0.01% of the full parameters.

### XBRL Datasets

#### Reporting Datasets

| Datasets        | Type    | Train/Test Split | Metrics      | Source                                                         |
|-----------------|---------|------------------|--------------|----------------------------------------------------------------|
| FiNER [7]       | Tagging | 900K / 100K      | Accuracy, F1 | [HF](https://huggingface.co/datasets/nlpaueb/finer-139?row=16) |
| FNXL [8]        | Tagging | 1K / 1K          | Accuracy, F1 | [GitHub](https://github.com/soummyaah/FNXL)                    |
| Warrant Tagging | Tagging | - / -            | Accuracy, F1 | -                                                              |
| Tag Query [10]  | Tagging | - / 50           | FActScore    | -                                                              |

#### Analysis Datasets

| Datasets             | Type        | Train/Test Split | Metrics   | Source                                                                                                                        |
|----------------------|-------------|------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------|
| Tags Extraction      | Extraction  | 300 / 150        | Accuracy  | -                                                                                                                             |
| Values Extraction    | Extraction  | 1K / 150         | Accuracy  | -                                                                                                                             |
| Formulas             | Extraction  | 300 / 150        | Accuracy  | -                                                                                                                             |
| Formula Calculations | Extraction  | 1K / 150         | Accuracy  | -                                                                                                                             |
| Financial Math [9]   | Math        | - / 1K           | Accuracy  | [GitHub](https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/formulas_with_explanations_with_questions_with_gt.xlsx) |
| Ratio Formulas [10]  | Math        | - / 50           | Accuracy  | -                                                                                                                             |
| XBRL Term [9]        | Terminology | - / 6K           | FActScore | [GitHub](https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/XBRL%20Terminology.xlsx)                                |
| Domain Query [9]     | QA          | - / 50           | FActScore | -                                                                                                                             |
| Numeric Query [9]    | QA          | - / 50           | FActScore | -                                                                                                                             |

### File Structures

```
FinLoRA
├── benchmarks
│   ├── run_evaluate.sh
│   ├── run_test_all.sh
│   ├── benchmarks.py
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
│   └── xbrl_extract.ipynb
│   ├── test
│   ├── train
├── environment.yml
└── src
    └── finetune
        ├── script_train.sh
        ├── train_lora.py
        └── utils.py
```
### Cross-task Generalization (LoRA MoE)

Currently we used single-task finetuning, i.e., finetune a LoRA adaptor for a task, and got higher performance. It is
practical for applications.

Mixture of LoRA Experts (LoRA-MoE): a LoRA module acts as an expert, a router network assigns weights. One
implementation is [X-LoRA](https://arxiv.org/pdf/2402.07148) [4]. X-LoRA is built on top of huggingface PEFT,
implementation should be easy.

### Improve Performance and Scalability for Inference

SLoRA [5] is designed for the serving of many LoRA adapters efficiently. It stores all adapters in the CPU memory and
fetches the adapters needed to GPU memory. We will deploy it on a cloud server.

Difficulty: Current SLoRA implementation does not work with HuggingFace, and does not support newer model like Llama 3.

### Distributed Training with Enhanced Privacy

Multiple institutions might want to collaborate for finetuning using their private data. Using zero-knowledge proof in
the finetuning stage allows enhanced privacy.


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

[6] Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, Matt White, Meikang Qiu,
Differentially Private Low-Rank Adaptation of Large Language Model Using Federated
Learning, https://arxiv.org/abs/2312.17493 ACM Transactions on Management Information Systems, 2024.

[7] Loukas, L.; Fergadiotis, M.; Chalkidis, I.; Spyropoulou, E.; Malakasiotis, P.; Androutsopoulos, I.;
and Paliouras, G. 2022. FiNER: Financial Numeric En
tity Recognition for XBRL Tagging. In Muresan, S.;
Nakov, P.; and Villavicencio, A., eds., Proceedings of
the 60th Annual Meeting of the Association for Compu
tational Linguistics (Volume 1: Long Papers). Dublin,
Ireland: Association for Computational Linguistics.

[8] Sharma, S.; Khatuya, S.; Hegde, M.; Shaikh, A.; Das-
gupta, K.; Goyal, P.; and Ganguly, N. 2023. Financial Numeric Extreme Labelling: A dataset and benchmarking. In Rogers,
A.; Boyd-Graber, J.; and Okazaki,
N., eds., Findings of the Association for Computational
Linguistics: ACL 2023, 3550–3561. Toronto, Canada:
Association for Computational Linguistics.

[9] Han, S.; Kang, H.; Jin, B.; Liu, X.-Y.; and Yang,
S. Y. 2024. XBRL Agent: Leveraging Large Language Models for Financial Report Analysis. In Proceedings of the 5th ACM
International Conference on
AI in Finance, ICAIF ’24, 856–864. New York, NY, USA: Association for Computing Machinery. ISBN
9798400710810.

[10] Wang, K.; Patel, J.; Shen, C.; Kim, D.; Zhu, A.; Lin,
A.; Borella, L.; Osborne, C.; White, M.; Yang, S.;
and Yanglet, K. X. X.-Y. L. 2024. A Report on
Financial Regulations Challenge at COLING 2025.
arXiv:2412.11159.
