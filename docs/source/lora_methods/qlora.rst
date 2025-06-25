Quantized LoRA (QLoRA)
~~~~~~~~~~~~~~~~~~~~~~

Motivation
----------

Vanilla LoRA fine-tuning requires substantial GPU memory, limiting accessibility for researchers with constrained resources. QLoRA [QLoRA]_ addresses this by combining LoRA's parameter efficiency with 4-bit quantization, enabling fine-tuning of large models on single consumer GPUs with significant memory reduction. For inference, QLoRA can also help 4-bit quantized model to restore the performance of 16-bit baseline, while saving GPU memory.

Technical Components
--------------------

**4-bit NormalFloat (NF4) Quantization**
QLoRA uses NF4, optimized for normally distributed weights. Unlike uniform quantization, NF4 allocates more bins near zero where weights cluster, providing better precision for critical values.

**Blockwise Quantization**
Weight tensors are divided into 64-element blocks, each quantized independently. This prevents precision loss from extreme values affecting the entire tensor.

**Double Quantization**
Quantization constants themselves are quantized to 8-bit, further reducing memory usage with minimal quality impact.

**Paged Optimizers**
Automatically move optimizer states between GPU and CPU memory to prevent out-of-memory errors during training.

The forward pass follows: :math:`\boldsymbol{y} = p_{16}(\boldsymbol{W}_0^{\text{NF4}}) \boldsymbol{x} + \gamma_r \boldsymbol{B} \boldsymbol{A} \boldsymbol{x}`

Using QLoRA in FinLoRA
----------------------

Enable QLoRA by setting ``quant_bits`` to ``4``:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_4bits_r4

Example configuration:

.. code-block:: json

   "sentiment_llama_3_1_8b_4bits_r4": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 4,
     "quant_bits": 4,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2
   }

**Key Parameters:**
- ``quant_bits``: Set to ``4`` for 4-bit quantization
- ``lora_r``: LoRA Rank
- ``learning_rate``: Often requires smaller values (0.0001-0.001)

QLoRA adapters are saved in ``lora_adapters/4bits_r4`` directory after fine-tuning.

.. [QLoRA] See the full citation in `references <../references>`