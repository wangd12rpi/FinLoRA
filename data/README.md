# FinLoRA Datasets

> The XBRL extraction datasets will be published soon.


### Training Dataset

Current the following datasets are included in the `train` directory, they are processed from their original
sources.

| Datasets                          | Type               | Train Size | Source                                                              |
|-----------------------------------|--------------------|------------|---------------------------------------------------------------------|
| Sentiment (FPB, FIQA, TFNS, NGWI) | Sentiment Analysis | 76.8K      | [HF](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) |
| Headline                          | Headline Analysis  | 82.2K      | [HF](https://huggingface.co/datasets/FinGPT/fingpt-headline-cls)    |
| NER                               | NER                | 13.5K      | [HF](https://huggingface.co/datasets/FinGPT/fingpt-ner-cls)         |
| FiNER                             | XBRL Tagging       | 900K       | [HF](https://huggingface.co/datasets/nlpaueb/finer-139?row=16)      |

### Testing Dataset

#### In `./test`

Current the following datasets are included in the `test` directory, they are also processed from the original
sources.

| Datasets | Type         | Train Size | Source                                                         |
|----------|--------------|------------|----------------------------------------------------------------|
| FiNER    | XBRL Tagging | 100K       | [HF](https://huggingface.co/datasets/nlpaueb/finer-139?row=16) |

#### Hosted on HuggingFace

The testing split of **FPB, FiQA, TFNS, NWGI, Headline, and NER** are not included in the GitHub repo. They will be
automatically downloaded from Huggingface when running testing code in `../test`.

