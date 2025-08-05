Dear reviewer,

Thank you for your prompt response and insightful question.

## Interpreting LoRA Interference via Task Similarity

To further measure the correlation between in-category task similarity and multi-task performance, we estimate the
in-category task similarity by averaging the cosine similarity between each pair of LoRA adapters within each task
category. The cosine similarity is computed by first calculating the effective weight update matrix ($ΔW=BA$) for each
attention layer in a LoRA adapter. These matrices are then flattened into vectors. For any two tasks, we compute
the cosine similarity between their corresponding vectors for each shared attention layer and then average these
layer-wise similarities. The final value in the table is the average of these pairwise similarities across all tasks
within the category.

The final task similarity and the average single-task and multi-task performance (and their difference) are displayed
below.

| Task Category                      | Avg. Cosine Similarity | Avg. Single-Task Performance | Avg. Multi-Task Performance | Avg. Performance $\Delta$ (Multi - Single) |
|:-----------------------------------|:----------------------:|:----------------------------:|:---------------------------:|:------------------------------------------:|
| General Financial Tasks            |         0.0043         |            81.97             |            54.75            |                 **-27.22**                 |
| Financial Reporting Tasks          |         0.0048         |            52.52             |            22.67            |                 **-29.85**                 |
| Financial Statement Analysis Tasks |       **0.0108**       |            74.17             |          **81.48**          |                 **+7.31**                  |

The results show a clear correlation between the similarity of LoRA adapters and the success of multi-task learning.

- Positive Transfer: The Financial Statement Analysis tasks show the highest average similarity (0.0108). This
  corresponds to successful multi-task learning, where the combined model outperformed the average of single-task models
  by 7.31 points. This suggests the tasks in this category are complementary, and their learned parameter updates are
  constructive.

- Negative Interference: The General Financial and Financial Reporting tasks and have low similarity
  scores (0.0043 and 0.0048). This dissimilarity leads to interference in the multi-task setting, causing a significant
  performance drop of -29.85 and -27.22 points compared to
  their single-task counterparts. This indicates that the parameter updates required for these tasks are conflicting.

## A Benchmark for Alleviating Interference

While our benchmark itself does not actively alleviate interference, Its structured allows it to serve as a
controlled environment for developing and validating methods that do.

To ensure we fully address your concern, could you please clarify what you mean by "provide unique problems to
alleviate the interference of LoRA?" For example, are you referring to creating a specific training dataset that
inherently
reduces interference? Your clarification would be invaluable for us to improve our work.

---

Thank you again for your valuable feedback. 

