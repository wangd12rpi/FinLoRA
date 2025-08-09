Dear Reviewer,

Thank you for your acknowledgement during the discussion period.

## Update on Q6: Domain-Shift at Larger Ranks

In the original rebuttal we showed the forgetting study with a single LoRA model, due to character limits.  
Below, we report the complete set of results.

We have also added the experiment you requested with a higher LoRA rank (rank 32).

| Benchmark          | Llama-3 8B Base | FiNER – LoRA (8-bit, r = 8) | FiNER – QLoRA (4-bit, r = 4) | **FiNER – QLoRA (4-bit, r = 32)** | FiNER – DoRA (8-bit, r = 8) | FiNER – rsLoRA (8-bit, r = 8) |
|:-------------------|:---------------:|:---------------------------:|:----------------------------:|:---------------------------------:|:---------------------------:|:-----------------------------:|
| MMLU               |      0.229      |            0.229            |            0.229             |               0.229               |            0.229            |             0.229             |
| GSM8K              |      0.011      |            0.011            |            0.014             |               0.010               |            0.011            |           **0.016**           |
| TriviaQA-Open (F1) |      0.667      |            0.658            |          **0.673**           |               0.666               |            0.663            |             0.663             |
| CoQA (F1)          |      0.711      |            0.709            |            0.697             |             **0.715**             |            0.707            |           **0.715**           |

This new result indicates that, even at larger ranks, LoRA introduce negligible
catastrophic forgetting: they preserve broad knowledge learned during pre-training while still enabling gains on the
in-domain financial tasks.

---

As we approach the end of the discussion period, we want to again express our sincere thanks for your guidance. Your
review has been instrumental in improving our paper, and we have worked carefully to address all of
your concerns to the best of our ability.

We hope our current responses have been satisfactory, and if so, we would be grateful if you would consider this in your
updated score. Please know that we welcome any further questions, and we are eager not only to answer them but also to
incorporate any further guidance to strengthen our work.

Thank you again for your time and valuable insights.

Sincerely,

The Authors

