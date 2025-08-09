Dear reviewer,

We are providing an update on the third question you previously stated: task categorization for multi-task fine-tuning.
We decided to use task similarity to
cluster-related tasks, creating a new categorization specifically for multi-task fine-tuning.

## An Update on Q3 Task Categorization

To do this, we used a sentence embedding method. For each dataset, we randomly sampled up to 200 training examples. We
then concatenated the question and answer for each example and used a sentence-transformer model to convert them into
embeddings. Finally, we calculated the average embedding for each dataset and computed the cosine similarity between all
dataset pairs. This process yields a value from 0 to 1, where a higher value indicates greater semantic similarity
between tasks.

### Task Similarity and Multi-task Performance

Below are the similarity scores for our original task categories.

#### General Category

|  | Sentiment Analysis | Headline |NER |
|:-------------------|-------------------:|---------:|------:|
| Sentiment Analysis |  1.000 | 0.556 | 0.244 |
| Headline  |  0.556 | 1.000 | 0.236 |
| NER |  0.244 | 0.236 | 1.000 |

#### Reporting Category

|  | XBRL-Term | FiNER |
|:----------|----------:|------:|
| XBRL-Term |  1.000 | 0.489 |
| FiNER  |  0.489 | 1.000 |

#### Analysis Category

| | Tag-Extraction | Financial-Math | FinanceBench |
|:---------------|---------------:|---------------:|-------------:|
| Tag-Extraction | 1.000 | 0.472 |  0.579 |
| Financial-Math | 0.472 | 1.000 |  0.644 |
| FinanceBench| 0.579 | 0.644 |  1.000 |

### Result Interpretation

We computed the average task similarity for dataset pairs within each original category. These averages align with our
multi-task LoRA evaluation results, suggesting that similarity is a reliable indicator for grouping datasets for joint
fine-tuning.

* **General (Avg. Similarity: 0.345) — Mixed or Negative Performance:** The low similarity score suggested potential
  task conflict. The performance of TFNS and Headline, which use different input styles, decreased significantly when
  trained together. NER performance also fell by 22 points. Only FiQA-SA improved slightly.

* **Reporting (Avg. Similarity: 0.489) — Mild Gains:** The similarity is moderate. Multi-task training helped FiNER (+4
  points) and XBRL-Term (+8 points, BERTScore), but the improvements were smaller than those in the Analysis category. (
  Note: Reporting results have been updated; please refer to Section 3).

* **Analysis (Avg. Similarity: 0.565) — Mostly Positive Gains:** The higher similarity appeared to facilitate knowledge
  transfer. Financial-Math performance jumped by 28 points, Formula-Construction by 6 points, and FinanceBench by 11
  points under multi-task tuning.

### Improved Task Clustering for Multi-task Fine-tuning

We computed the complete 19x19 similarity matrix across all datasets (the full matrix is omitted for brevity).
Clustering this matrix allowed us to form new, more cohesive groups for multi-task fine-tuning.

| Cluster  | Tasks (Members) | Mean Similarity |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|
| 1. News & Sentiment  | FiQA, FPB, NWGI, TFNS, Headline| **0.769** |
| 2. NER & Certification  | CFA-1, CFA-2, CFA-3, CPA-REG, NER | **0.580** |
| 3. Financial Statement Reasoning | FinanceBench, FiNER, FNXL, Financial-Math, XBRL-Formula-Calculation, XBRL-Formula-Construction, XBRL-Tag-Extraction, XBRL-Val-Extraction, XBRL-Term | **0.613** |

The mean similarity within these new clusters is significantly higher than in our original categories, and we expect
less negative interference during multi-task fine-tuning. We plan to conduct a new round of fine-tuning based on this
new grouping and will report the evaluation results in the camera-ready version.

---

As we approach the end of the discussion period, we want to again express our sincere thanks for your thorough
engagement. Your feedback has been instrumental in improving our paper, and we have worked carefully to address all of
your concerns.

We hope our current responses have been satisfactory, and if so, we would be grateful if you would consider this in your
updated score. Please know that we welcome any further questions, and we are eager not only to answer them but also to
incorporate any further guidance to strengthen our work.

Thank you again for your time and valuable insights.

Sincerely,

The Authors
