# Append the following after the last paragraph in the section `## Expanded Model Diversity` in the summary

Furthermore, we have begun evaluating and fine-tuning the newly released **`GPT-oss-20B`** model. Due to the tight
rebuttal timeframe, we have only completed the fine-tuning for the sentiment analysis task so far. The initial results
are...

| **Datasets** | Llama 3.1 8B | _GPT-oss-20B_ | **Llama 3.1 8B (3 shot)** | _GPT-oss-20B (3 shot)_ | Llama 3.1 8B LoRA 8bit-r8 | _GPT_oss-20B (Fine-tuned)_ |
|:-------------|:------------:|---------------|:-------------------------:|------------------------|:-------------------------:|----------------------------|
| FPB          |    68.73     |               |           76.40           |                        |         **85.64**         |                            |
| FiQA SA      |    46.55     |               |           64.68           |                        |           81.28           |                            |
| TFNS         |    69.97     |               |           28.81           |                        |         **88.02**         |                            |
| NWGI         |    43.86     |               |           32.20           |                        |         **54.16**         |                            |

