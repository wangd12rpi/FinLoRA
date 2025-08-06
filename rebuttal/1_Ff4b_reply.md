Dear Reviewer,

Thank you for your insightful questions.

### Regarding Q1: Clarification on LoRA Adapters and Task Categorization

You are correct that our "Financial Certification" category includes both the Analyst mock exam (CFA) and the regulation
section of the Accountant mock exam (CPA REG). However, the dataset we used has a focus on regulations/ethics,
and their formats are nearly identical (multiple-choice questions). Due to the subject and structural similarity, we
consider them a single task, hence a single LoRA adapter. We apologize for the lack of clarity, and we will make sure to
emphasize this in the camera-ready version.

The table below clarifies the 9 single-task LoRA adapters used in our study.

| Category| Task(s) | # Adapters |
|:-----------------------------|:--------------------------------------------|:----------:|
| General Finance  | Sentiment Analysis, NER, Headline Analysis  |  3|
| Financial Certification| Certification (combined Analyst/Accountant) |  1|
| Financial Reporting | FiNER, FNXL, XBRL Term |  3|
| Financial Statement Analysis | XBRL Analysis, FinanceBench, Financial Math |  2|
| **Total**  ||**9** |

### Regarding Q2: Multi-task Performance on Financial Reporting Tasks

Thank you for highlighting this anomaly. Your question led us to discover a configuration error made when running our
rebuttal experiments. Under the tight time constraints during rebuttal, we incorrectly set the delimiter parameter that
parses the output for the FiNER and FNXL evaluation, leading to the low scores. We apologize for this mistake in
execution and present the corrected results and updated average below:

| **Datasets** | Llama 3.1 8B (Base) | **Llama 3.1 8B (3-shot)** | Llama 3.1 8B LoRA (Single-task) | **Llama 3.1 8B LoRA (Multi-task)** |
|:----------------------------|:-------------------:|:-------------------------:|:-------------------------------:|:----------------------------------:|
| FiNER  |  21.28  |  30.76  |**74.10**|67.97 |
| FNXL|  03.64  |  13.15  |  23.57  | **28.23**  |
| XBRL Term (BERTScore) |  0.574  |  0.595  |  0.599  | **0.676**  |
| **Updated Overall Average** |  37.05  |  47.38  |**74.74**|68.78 |

The corrected results are significantly higher. But there is still no clear advantage for multi-task training, at least
for 0-shot inference, as the task formats differ significantly:

* **FiNER/FNXL**: NER-style classification (assigning a GAAP tag to a number in a sentence).
* **XBRL Term**: Definitional task (generating a tag's definition).

As requested, we also evaluated the model fine-tuned *solely* on XBRL Term on the FiNER and FNXL tasks.

| **Datasets** | **Llama 3.1 8B LoRA (0-shot)** | **Llama 3.1 8B LoRA (3-shot)** |
|:-------------|:------------------------------:|:------------------------------:|
| FiNER  | 20.79  | 32.77  |
| FNXL| 05.24  | 18.88  |

The fine-tuned 0-shot performance is similar to 0-shot base. However, the fine-tuned 3-shot performance outperforms
3-shot base, confirming that the knowledge learned for the definitional XBRL Term task does transfer to the NER-style
tagging format during the 3-shot settings where the model familiarizes with the correct answer format.

For future work, we hypothesize that we could improve multi-task
performance by enhancing the FiNER/FNXL training data using XBRL Term dataset. For instance, we could inject the
definition of the correct tag (from XBRL Term) into the training examples answers. This would teach the model the
reasoning behind each tag choice. We plan to explore this for the camera-ready version.

### Regarding Q3: Rationale for Task Categorization

We based our task categories on financial domain application to provide an intuitive framework for the finance sector.
For example, financial reporting tasks refer to the set of work that a company does when they prepare to file an annual
report with the SEC.
However, we acknowledge this conceptual grouping doesn't guarantee positive transfer during multi-task training. As our
results show, disparate task formats within a single category can lead to negative transfer.

To address this for the camera-ready version, we will introduce a discussion on alternative groupings to guide readers
on which tasks are best suited for multi-task fine-tuning. Furthermore, we will conduct more comprehensive multi-task
experiments and release the resulting high-performing merged datasets on Hugging Face.

Thank you again for your constructive feedback.

Sincerely,

The Authors