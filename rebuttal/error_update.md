We discovered a configuration error made when running our multi-task
rebuttal experiments on financial reporting tasks. Under the tight time constraints during rebuttal, we incorrectly set the delimiter parameter that
parses the output for the FiNER and FNXL evaluation, leading to the low scores. We apologize for this mistake in
execution and present the corrected results and updated average below:

| **Datasets** | Llama 3.1 8B (Base) | **Llama 3.1 8B (3-shot)** | Llama 3.1 8B LoRA (Single-task) | **Llama 3.1 8B LoRA (Multi-task)** |
|:----------------------------|:-------------------:|:-------------------------:|:-------------------------------:|:----------------------------------:|
| FiNER  |  21.28  |  30.76  |**74.10**|67.97 |
| FNXL|  03.64  |  13.15  |  23.57  | **28.23**  |
| XBRL Term (BERTScore) |  0.574  |  0.595  |  0.599  | **0.676**  |
| **Updated Overall Average** |  37.05  |  47.38  |**74.74**|68.78 |

The corrected FiNER and FNXL results are significantly higher. But there is still no clear advantage for multi-task training for financial reporting tasks, at least for 0-shot inference, as the task formats differ significantly:

* **FiNER/FNXL**: NER-style classification (assigning a GAAP tag to a number in a sentence).
* **XBRL Term**: Definitional task (generating a tag's definition).

Sincerely,

The Authors
