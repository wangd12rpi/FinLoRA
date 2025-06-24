Vanilla LoRA
============================



LoRA (Low-Rank Adaptation) is a technique that makes fine-tuning large language models more efficient and practical. Instead of updating all parameters in a model during fine-tuning, LoRA provides an approach that updates only a small subset of parameters while keeping most of the model frozen.


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


When to Use LoRA
----------------

LoRA is particularly valuable when:

* You need multiple specialized versions of a model
* Quick adaptation to new domains or tasks is required
* You prefer low computational cost
* You still want to keep world knowledge of the base model

Using LoRA in FinLoRA
----------------------

To use standard LoRA in FinLoRA, you can configure fine-tuning without specifying any special parameters for other LoRA variants. Here's an example of how to configure standard LoRA for fine-tuning:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8

This uses the configuration from ``lora/finetune_configs.json``:

.. code-block:: json

   "sentiment_llama_3_1_8b_8bits_r8": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 8,
     "quant_bits": 8,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2
   }

The key parameters for standard LoRA are:
- ``lora_r``: The rank of the LoRA adapter
- ``quant_bits``: The quantization bits (8 for standard LoRA)
- ``lora_alpha``: The scaling factor for the LoRA adapter (optional, default is 16)

Standard LoRA adapters are saved in the ``lora_adapters/8bits_r8`` directory after fine-tuning.

References
----------

.. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.

.. [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.

.. [3] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024, July). Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

.. [4] Kalajdzievski, D. (2023). Rank-stabilized scaling factor for LoRA adaptation.

.. [5] Liu, X. Y., Zhu, R., Zha, D., Gao, J., Zhong, S., White, M., & Qiu, M. (2025). Differentially private low-rank adaptation of large language model using federated learning. ACM Transactions on Management Information Systems, 16(2), 1-24.
```
