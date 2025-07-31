Dear NeurIPS Reviewer:

Thank you for the constructive suggestions.

## Q1 How does LoRA alleviate the issues highlighted in Table1?

| Error class  | Error in Table1| Why the base Llama-3-8B fails| How LoRA fixes it  |
|----------------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **1. Unfamiliarity with US GAAP tags and Hallucination** | Mis-tags \$2bn as `MajorityEquityInterest` | `MajorityEquityInterest` is not part of the US GAAP taxonomy. The base model hallucinated and created an non-existing tag.   | After seeing the US GAAP tags during the fine-tuning process, the model know the valid tag name.   |
| **2. Unfamiliarity with financial formula**  | Picks 1,209,000,000 instead of 125,978,000,000 | Base model did not know the formula of equality multiplier. `EquityMultiplier = Assets / Equity`, therefore did not select the correct assets value. | Fine-tuning exposed the LLM with relevant financial concepts and formula, allowing it to select the correct value. |

Overall, LoRA succeeds because fine-tuning exposed the model with financial concepts that the base model does not have.

## Q2 What are the most difficult points in financial data?

Financial data poses three main challenges:

1. Domain-specific vocabulary: Many specialized terms rarely appear in general corpora, so the base LLM has limited
   exposure and often misinterprets them.

2. Mixed formats: A single report can contain narrative text, XML tags, tables, and raw numbers. The model must process
   information across these varied structures.

3. Long context windows: XBRL documents remain lengthy even after excessive texts are removed. For example, in XBRL
   Analysis, inputs often reach about four thousand tokens, and numbers required by the same formula may be thousands of
   tokens apart.

## Q3 Future research directions in fine-tuning: what drives adaptation performance?

We selected these three future directions to investigate the core drivers of fine-tuning performance. 
In our original submission we only tested single task fine-tuning. We are adding
multi-task fine-tuning results for rebuttal, and our initial results displayed a trade-off between single- and
multi-task learning, where some tasks benefited while others suffered from negative transfer. This made investigating
the single- vs. multi-task dynamic a starting point. The observed interference led to our second
direction: exploring adapter capacity. We believe that small LoRA capacity (i.e., a low LoRA rank) may causes those
lower performance in multi-task settings. Finally, to manage these challenges, we also hope
 to explore cross-task routing mechanisms like Mixture of LoRA Experts (MoLE). If multi-tasking is suboptimal,
expert selection might further mitigate interference and display the actual benefits of joint training.

### Single-task vs. multi-task fine-tuning

We performed additional experiments to compare single-task and multi-task LoRA fine-tuning and added an additional 3
shot baseline. The detailed results are summarized below:

| **Datasets**   | Llama 3.1 8B | **Llama 3.1 8B (3 shot)** | Llama 3.1 8B LoRA 8bit-r8 (Single-task) | **Llama 3.1 8B LoRA 8bit-r8 (Multi-task)** |
|:---------------------------------------|:------------:|:-------------------------:|:---------------------------------------:|:------------------------------------------:|
| **General Financial Tasks**|  |   | ||
| FPB|68.73 |   76.40   |**85.64**|   85.31|
| FiQA SA|46.55 |   64.68   |  81.28  | **82.20**  |
| TFNS   |69.97 |   28.81   |**88.02**|   34.51|
| NWGI   |43.86 |   32.20   |**54.16**|   36.51|
| NER|48.89 |   55.34   |**98.05**|   76.07|
| Headline   |45.34 |   70.01   |**84.66**|   13.90|
| **Financial Certification Tasks**  |  |   | ||
| CFA 1  |13.33 |   51.11   |**86.67**| -  |
| CFA 2  |19.48 |   37.66   |**88.31**| -  |
| CFA 3  |16.67 |   51.28   |**70.51**| -  |
| CPA|31.68 |   45.54   |**80.20**| -  |
| **Financial Reporting Tasks**  |  |   | ||
| FiNER  |21.28 |   30.76   |**74.10**|0.41|
| FNXL   | 3.64 |   13.15   |**23.57**|0.00|
| XBRL Term (BERTScore)  |0.574 |   0.595   |  0.599  | **0.676**  |
| **Financial Statement Analysis Tasks** |  |   | ||
| Tag Extraction |69.16 |   70.22   |**89.13**|   88.78|
| Value Extraction   |52.46 |   72.27   |**98.49**|   97.62|
| Formula Construction   |12.92 |   17.73   |  77.61  | **83.33**  |
| Formula Calculation|27.27 |   33.65   |  98.68  | **99.04**  |
| Finance Bench (BERTScore)  |0.443 |   0.580   |  0.511  | **0.621**  |
| Financial Math |11.00 |   32.00   |  30.00  | **58.00**  |
| **Overall Average**|  |   | ||
| Aggregated |37.05 |   47.38   |**74.74**|   63.74|

*Due to character limit, the table only lists Accuracy. For BERTScore we report the F1 value.*

_For financial certification tasks, there is no multi-task score as we consider them to be only one task due to similar
format_

Multi-task fine-tuning produces clear gains in Financial Statement Analysis. Formula construction, formula
calculation, Finance Bench, and financial math improve under the multi-task setting. These tasks share similar
underlying knowledge like the structure of financial statements and numerical reasoning. Learning them together
helps the model to achieve an enhanced understanding in these topics.

In contrast, we see negative transfer in General Financial and Financial Reporting tasks, where the multi-task model
performs worse on TFNS, Headline, FiNER, and FNXL. We suspect the problem comes from differences in task format and
objective. Even with balanced sampling, the model struggles to optimize for very different objectives at the same time.

Overall performance drops slightly when tasks are merged. Closely related tasks can benefit from joint training, while
divergent tasks often harm each other.

We plan to extend the multi-task experiment to other LoRA methods as well.

### Adapter capacity: lora rank vs task complexity, how did it affect performance?

We compared LoRA ranks of 4 and 8 and observed rank 8 provided higher accuracy across most tasks. However, certain
particularly challenging tasks, especially under multi-task fine-tuning, may benefit from even higher LoRA ranks (e.g.,
16 or 32). We plan to benchmark LoRA ranks above 8 to identify optimal configurations for complex scenarios.

### Cross‑task interference and routing

Currently, adapters are selected manually according to task types. However, we plan to adopt Mixture‑of‑LoRA‑Experts (
MoLE) routing which dynamically selects adapters via a small gating network. We are running a test on such method and
will append a brief result to supplementary materials for camera‑ready version.


## Q4 Additional Catastrophic Forgetting Experiments

We have extended our “out-of-domain” suite beyond MMLU and GSM8K to additional general-knowledge benchmarks:

| Benchmark | Llama 3.1 8B Base | Llama 3.1 8B Fine-tuned for FiNER |
|---------------|:-----------------:|:---------------------------------:|
| MMLU  |       0.229       |               0.229               |
| GSM8K |       0.011       |               0.011               |
| TriviaQA-Open |       0.667       |               0.663               |
| CoQA  |       0.711       |               0.709               |

Overall, we find minimal signs of catastrophic forgetting after the fine-tuning process.

---

We appreciate your feedback. Your comments led to a stronger benchmark.
