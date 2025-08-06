Dear NeurIPS Reviewer:

Thank you for the constructive suggestions.

## Q1 How does LoRA alleviate the issues highlighted in Table1?

| Error class                                              | Error in Table1                                | Why the base Llama-3-8B fails                                                                                                                          | How LoRA fixes it                                                                                                  |
|----------------------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **1. Unfamiliarity with US GAAP tags and Hallucination** | Mis-tags \$2bn as `MajorityEquityInterest`     | `MajorityEquityInterest` is not part of the US GAAP taxonomy. The base model hallucinated and created a non-existent tag.                              | After seeing the US GAAP tags during the fine-tuning process, the model knows the valid tag name.                  |
| **2. Unfamiliarity with financial formula**              | Picks 1,209,000,000 instead of 125,978,000,000 | Base model did not know the formula of the equity multiplier. `EquityMultiplier = Assets / Equity`, therefore did not select the correct assets value. | Fine-tuning exposed the LLM with relevant financial concepts and formula, allowing it to select the correct value. |

Overall, LoRA succeeds because fine-tuning exposed the model to financial concepts that the base model does not have.

## Q2 What are the most difficult points in financial data?

Financial data poses three main challenges:

1. Domain-specific vocabulary: Many specialized terms rarely appear in general corpora, so the base LLM has limited
   exposure and often misinterprets them.

2. Mixed formats: A single report can contain narrative text, XML tags, tables, and raw numbers. The model must process
   information across these varied structures.

3. Long context windows: XBRL documents remain lengthy even after excessive text is removed. For example, in XBRL
   Analysis, inputs often reach about four thousand tokens, and numbers required by the same formula may be thousands of
   tokens apart.

## Q3 Future research directions in fine-tuning: what drives adaptation performance?

We selected these three future directions to investigate the core drivers of fine-tuning performance.
In our original submission we only tested single task fine-tuning. We are adding
multi-task fine-tuning results for rebuttal, and our initial results displayed a trade-off between single- and
multi-task learning, where some tasks benefited while others suffered from negative transfer. This made investigating
the single- vs. multi-task dynamic a starting point. The observed interference led to our second
direction: exploring adapter capacity. We believe that small LoRA capacity (i.e., a low LoRA rank) may cause the
lower performance in multi-task settings. Finally, to manage these challenges, we also hope
to explore cross-task routing mechanisms like Mixture of LoRA Experts (MoLE). If multi-tasking is suboptimal,
expert selection might further mitigate interference and reveal the actual benefits of joint training.

### Single-task vs. multi-task fine-tuning

We performed additional experiments to compare single-task and multi-task LoRA fine-tuning and added an additional 3
shot baseline. The detailed results are summarized below:

[see table on ff4b]

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

Overall, performance drops by 11 points when tasks are merged. Closely related tasks can benefit from joint training,
while
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

| Benchmark     | Llama 3.1 8B Base | Llama 3.1 8B Fine-tuned for FiNER |
|---------------|:-----------------:|:---------------------------------:|
| MMLU          |       0.229       |               0.229               |
| GSM8K         |       0.011       |               0.011               |
| TriviaQA-Open |       0.667       |               0.663               |
| CoQA          |       0.711       |               0.709               |

Overall, we find minimal signs of catastrophic forgetting after the fine-tuning process.

---

We appreciate your feedback. Your comments led to a stronger benchmark.

