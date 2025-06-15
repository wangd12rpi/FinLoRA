LoRA Methods
============================



What is LoRA?
-------------
LoRA (Low-Rank Adaptation) is a technique that makes fine-tuning large language models more efficient and practical. Instead of updating all parameters in a model during fine-tuning, LoRA provides an approach that updates only a small subset of parameters while keeping most of the model frozen.


Why LoRA - Traditional Fine-Tuning Challenges
---------------------------------------------

When working with LLMs, traditional full fine-tuning presents several challenges:

* **Resource Intensive**: Updating all parameters (which can be billions) requires substantial computational resources
* **Storage Problems**: Each fine-tuned model requires a complete copy of all parameters

For example, imagine a pre-trained model with 500 million parameters. With traditional fine-tuning, you'd need to update all 500 million parameters for each new task, which is extremely inefficient.

How LoRA Works
-------------------------------------

.. raw:: html

    <object class="figure" data="../_static/images/lora_diagram.png" type="image/svg+xml"></object>
    <br>




Here is a sample explanation of how LoRA works:

1. **Keep the Base Model Frozen**: Don't change any of the original model parameters
2. **Add a Second Channel**: Introduce small, trainable matrices that work alongside the original model
3. **Use Low-Rank Matrices**: These matrices are specifically designed to be very small but still capture important adaptations

The additional parameters required are minimal - typically less than 1% of the original model size.


Practical Example
~~~~~~~~~~~~~~~~

Let's say you have a weight matrix in your language model that is 1024×1024 (over 1 million parameters):

* With traditional fine-tuning: You would update all 1,048,576 parameters
* With LoRA (rank=8): You only update about 16,384 parameters (about 1.5% of the original)

During inference, the low-rank update can be merged with the original weight matrix, resulting in no additional computational overhead compared to the original model.

Benefits of Using LoRA
----------------------

1. **Efficiency**: Train significantly fewer parameters (typically <1% of the original model)
2. **No Overhead During Inference**: LoRA updates can be merged with the original weights for deployment
3. **Adaptability**: Create multiple specialized versions of your model for different tasks
4. **Storage**: Store one base model and multiple small LoRA adaptations instead of many complete models

LoRA Methods and Variants
-------------------------

This section introduces several LoRA methods and variants that have been developed to address different aspects of efficient fine-tuning.

Low-Rank Adaptation (LoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The original LoRA method adds trainable low-rank matrices to the frozen pre-trained model. These matrices create an update channel that can be added to the original weights.

For each weight matrix in the model (particularly in attention layers), LoRA introduces two smaller matrices that, when multiplied together, produce an update that gets added to the original weight.

During fine-tuning, only these smaller matrices are updated, while the original model remains frozen. This dramatically reduces the number of trainable parameters.

QLoRA
~~~~~

QLoRA combines LoRA with 4-bit quantization to further reduce memory requirements. By using lower precision numbers to represent the frozen model weights, QLoRA makes it possible to fine-tune very large language models on consumer hardware with limited GPU memory.

Key benefits include:
* Enables fine-tuning on consumer-grade hardware
* Maintains performance comparable to full-precision fine-tuning
* Further reduces memory requirements beyond standard LoRA

DoRA (Weight-Decomposed Low-Rank Adaptation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DoRA extends LoRA by decomposing weights into magnitude and direction components. This decomposition allows for more fine-grained updates while maintaining efficiency.

The key insight of DoRA is that separating magnitude and direction allows for more expressive adaptation with the same parameter budget, leading to improved performance on downstream tasks.

rsLoRA (Rank-Stabilized LoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rsLoRA introduces a scaling factor to improve gradient stability during training. This helps prevent training instability, especially in deeper models, by ensuring better gradient flow through the network.

This variant is particularly useful when working with very deep models where traditional LoRA might encounter optimization challenges.

Federated LoRA
~~~~~~~~~~~~~~

Federated LoRA combines LoRA with federated learning principles, allowing multiple organizations to collaboratively fine-tune models while keeping their data private.

This approach is particularly valuable in domains like finance and healthcare where data privacy is crucial but model improvement benefits from diverse data sources.

When to Use LoRA
----------------

LoRA is particularly valuable when:

* You need multiple specialized versions of a model
* Quick adaptation to new domains or tasks is required
* You prefer low computational cost
* You still want to keep world knowledge of the base model

References
----------

.. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.

.. [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.

.. [3] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024, July). Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

.. [4] Kalajdzievski, D. (2023). Rank-stabilized scaling factor for LoRA adaptation.

.. [5] Liu, X. Y., Zhu, R., Zha, D., Gao, J., Zhong, S., White, M., & Qiu, M. (2025). Differentially private low-rank adaptation of large language model using federated learning. ACM Transactions on Management Information Systems, 16(2), 1-24.
```