_**reviewer comments: **
Thank you for the additional experiments and analysis.

I have few questions based on the added results.

1. For financial certification tasks, doesn't it also include two categories, Analyst Exam and Accountant Exam, based on
   the Table 2? Otherwise, shouldn't it will be 8 LoRA adapter instead of 9 based on the task type?

2. Does the results within the Financial Reporting Task indicate adding the training data of XBRL Term will make the
   fine-tuned model perform even worse than the original model? Can the authors also show the zero/three-shots
   performance of the model solely fine-tuned on XBRL Term, on the FiNER and FNXL tasks?

3. If the adding the training data from some tasks will decrease the overall performance within the task category, what
   is the meaning to include them in the same category?

-------------------------------
-------------------------------
Dear Reviewer,

Thank you for your insightful questions.

### Regarding Q1: Clarification on LoRA Adapters and Task Categorization

You are correct that our "Financial Certification" category includes both the Analyst (CFA) and Accountant (CPA) exams.
While their subjects differ, their mock exam formats are nearly identical (multiple-choice questions). Due to this
structural similarity, we merged them into one "Certification" task for fine-tuning, hence a single LoRA adapter.

The table below clarifies the 9 single-task LoRA adapters used in our study.

| Category                     | Task(s)                                     | # Adapters |
|:-----------------------------|:--------------------------------------------|:----------:|
| General Finance              | Sentiment Analysis, NER, Headline Analysis  |     3      |
| Financial Certification      | Certification (combined Analyst/Accountant) |     1      |
| Financial Reporting          | FiNER, FNXL, XBRL Term                      |     3      |
| Financial Statement Analysis | XBRL Analysis, FinanceBench, Financial Math |     2      |
| **Total**                    |                                             |   **9**    |

### Regarding Q2: Multi-task Performance on Financial Reporting Tasks

Thank you for highlighting this anomaly. Your question led us to discover a configuration error made when running our
rebuttal experiments. Under the tight time constraints during rebuttal, we incorrectly set the delimiter parameter that
parses the output for the FiNER and FNXL evaluation, leading to the low scores. We apologize for this mistake in
execution and present the corrected results below:

| **Datasets**                | Llama 3.1 8B (Base) | **Llama 3.1 8B (3-shot)** | Llama 3.1 8B LoRA (Single-task) | **Llama 3.1 8B LoRA (Multi-task)** |
|:----------------------------|:-------------------:|:-------------------------:|:-------------------------------:|:----------------------------------:|
| FiNER                       |        21.28        |           30.76           |            **74.10**            |               67.97                |
| FNXL                        |        03.64        |           13.15           |              23.57              |             **28.23**              |
| XBRL Term (BERTScore)       |        0.574        |           0.595           |              0.599              |             **0.676**              |
| **Updated Overall Average** |        37.05        |           47.38           |            **74.74**            |               68.78                |

The corrected results are significantly higher. But there is still no clear advantage for multi-task training, at least
for 0-shot inference, as the task formats differ significantly:

* **FiNER/FNXL**: NER-style classification (assigning a GAAP tag to a number in a sentence).
* **XBRL Term**: Definitional task (generating a tag's definition).

As requested, we also evaluated the model fine-tuned *solely* on XBRL Term on the FiNER and FNXL tasks.

| **Datasets** | **Llama 3.1 8B LoRA (0-shot)** | **Llama 3.1 8B LoRA (3-shot)** |
|:-------------|:------------------------------:|:------------------------------:|
| FiNER        |             20.79              |             32.77              |
| FNXL         |             05.24              |             18.88              |

The fine-tuned 0-shot performance is similar to 0-shot base. However, the fine-tuned 3-shot performance outperforms
3-shot base, confirming that the knowledge learned for the definitional XBRL Term task does transfer to the NER-style
tagging format during the 3-shot settings where the model familiarizes with the correct answer format.

For future work, we hypothesize that we could improve multi-task
performance by enhancing the FiNER/FNXL training data using XBRL Term dataset. For instance, we could inject the
definition of the correct tag (from XBRL Term) into the training examples answers. This would teach the model the
reasoning behind each tag choice. We plan to explore this for the camera-ready version.

### Regarding Q3: Rationale for Task Categorization

Our task categories are based on **financial domain application**
to create an intuitive framework for the finance sector.

We acknowledge this conceptual grouping does not guarantee positive transfer for multi-task training. As shown, tasks
within a category can have disparate formats, leading to negative transfer.

For the camera-ready version, we will add a discussion on an alternative grouping to guide readers on
which tasks are best suited for multi-task fine-tuning.

Thank you again for your constructive feedback.

Sincerely,

The Authors_