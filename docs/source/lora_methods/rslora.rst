Rank-Stabilized LoRA (rsLoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LoRA scales the weight matrix update :math:`\boldsymbol{BA}` by :math:`\frac{\alpha}{r}`, which can cause gradients to explode or diminish as the rank :math:`r` increases. In contrast, rsLoRA uses a scaling factor :math:`\frac{\alpha}{\sqrt{r}}`:

.. math::

   \boldsymbol W'=\boldsymbol W_0+\frac{\alpha}{\sqrt{r}}\boldsymbol B\boldsymbol A.

This scaling results in gradient-scale stability at higher ranks, enabling the rank to be higher to capture more details in long-context tasks like XBRL extraction. rsLoRA also results in lower perplexity—the model assigns higher probabilities to correct words—than LoRA at higher ranks.

Using RSLoRA in FinLoRA
----------------------

To use RSLoRA in FinLoRA, you need to set the ``peft_use_rslora`` parameter to ``true`` in your configuration. Here's an example of how to configure RSLoRA for fine-tuning:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8_rslora

This uses the configuration from ``lora/finetune_configs.json``:

.. code-block:: json

   "sentiment_llama_3_1_8b_8bits_r8_rslora": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 8,
     "quant_bits": 8,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2,
     "peft_use_rslora": true
   }

The key parameters for RSLoRA are:
- ``peft_use_rslora``: Set to ``true`` to enable RSLoRA
- ``lora_r``: The rank of the LoRA adapter (RSLoRA works well with higher ranks)

RSLoRA adapters are saved in the ``lora_adapters/8bits_r8_rslora`` directory after fine-tuning.
