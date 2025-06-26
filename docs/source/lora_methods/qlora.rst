QLoRA
======

.. contents:: Table of Contents

Background
----------

**Citation:** `QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) <https://arxiv.org/abs/2305.14314>`_

QLoRA addresses the issue of LoRA fine-tuning still requires substantial GPU memory, which becomes prohibitive for very large LLMs. QLoRA combines LoRA's parameter efficiency with 4-bit quantization, enabling fine-tuning of 65B parameter models on a single 48GB GPU while preserving full 16-bit fine-tuning task performance. It can reduce the average memory requirements from >780GB to <48GB without degrading performance.

Quick Facts
~~~~~~~~~~~

#. QLoRA is a memory-efficient fine-tuning method that uses 4-bit quantization.
#. QLoRA introduces no decrease in performance compared to the full 16-bit LoRA fine-tuning.
#. QLoRA works with any neural network containing dense layers.

Algorithmic Idea
~~~~~~~~~~~~~~~~

The core idea behind QLoRA is that LoRA fine-tuning can be further optimized by quantizing the base model weights to 4 bits while keeping the adapter weights at full precision. This enables more people to fine-tune large models on their own hardware. QLoRA quantizes the pre-trained model weights to reduce memory usage while preserving the trainable LoRA parameters at 16-bit precision.

For a pre-trained weight matrix :math:`\mathbf{W}_0`, QLoRA quantizes it to 4-bit NormalFloat format. During computation, the quantized weights are dynamically dequantized back to 16 bits when performing operations with the input sequence :math:`\mathbf{x}` and the adapter matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`, which remain in 16-bit precision.

During fine-tuning, the following hold true:

#. :math:`\mathbf{W}_0` is quantized to 4-bit precision and frozen and doesn't receive any gradient updates.
#. Only :math:`\mathbf{A}` and :math:`\mathbf{B}` contain trainable parameters in 16-bit precision.
#. The forward pass dynamically dequantizes weights: :math:`\mathbf{h} = p_{16}(\mathbf{W}_0^{\text{NF4}}) \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}`.
#. QLoRA uses NF4 quantization and paged optimizers to reduce memory usage dynamically.

Key Equations
~~~~~~~~~~~~

For a pre-trained weight matrix :math:`\mathbf{W}_0`, the QLoRA forward pass is defined as:

.. math::

   \mathbf{y} = p_{16}(\mathbf{W}_0^{\text{NF4}}) \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}

Where:

#. :math:`p_{16}(\mathbf{W}_0^{\text{NF4}})` represents the dynamic dequantization of 4-bit NormalFloat weights back to 16-bit precision.
#. :math:`\mathbf{W}_0^{\text{NF4}}` are the pre-trained weights quantized to 4-bit NormalFloat format.
#. :math:`\mathbf{A} \in \mathbb{R}^{r \times k}` and :math:`\mathbf{B} \in \mathbb{R}^{d \times r}` are the 16-bit LoRA adapter matrices.
#. :math:`\gamma_r = \alpha/r` is the scaling factor where :math:`\alpha` is a hyperparameter and :math:`r` is the adapter rank.

The double dequantization process is:

.. math::

   \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{k\text{-bit}}) = \text{dequant}(\text{dequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}), \mathbf{W}^{4\text{-bit}})

Where :math:`c_1` and :math:`c_2` are the first and second-level quantization constants.

Implementation in FinLoRA
~~~~~~~~~~~~~~~~~~~~~~~~

To use QLoRA in FinLoRA, configure fine-tuning with 4-bit quantization:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_4bits_r4

Configuration example from ``lora/finetune_configs.json``:

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

Key parameters:
- ``lora_r``: The rank :math:`r` of the LoRA adapter (typically 4-8 for QLoRA)
- ``quant_bits``: The quantization bits (4 for QLoRA, automatically enables NF4 and optimizations)
- ``lora_alpha``: The scaling parameter :math:`\alpha` (default: 16, giving :math:`\gamma_r = \alpha/r`)

Usage Example
~~~~~~~~~~~~

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel
   import torch

   # Load base model
   base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       torch_dtype=torch.float16,
       device_map="auto"
   )

   # Load QLoRA adapter
   adapter_path = "./lora_adapters/4bits_r4/sentiment_llama_3_1_8b_4bits_r4"
   model = PeftModel.from_pretrained(base_model, adapter_path)

   # Generate text
   tokenizer = AutoTokenizer.from_pretrained(base_model_name)
   prompt = "The financial markets showed positive sentiment today"
   inputs = tokenizer(prompt, return_tensors="pt")
   
   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=100, temperature=0)
   
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)

References
----------

.. [1] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*.

Why This Method?
~~~~~~~~~~~~~~~

QLoRA is important for understanding memory-efficient fine-tuning of large language models that are too large to fit on a single GPU. It introduces key quantization techniques that enable LoRA fine-tuning to be done on consumer hardware without losing performance. QLoRA provides practical innovations for 4-bit fine-tuning that make fine-tuning accessible to a wider range of researchers at an affordable cost.

Useful Links
~~~~~~~~~~~~

* `Official QLoRA Implementation <https://github.com/artidoro/qlora>`_ - Original implementation by the authors
* `Hugging Face PEFT Documentation <https://huggingface.co/docs/peft/main/en/developer_guides/quantization>`_ - Official quantization guide for PEFT
* `BitsAndBytes <https://github.com/TimDettmers/bitsandbytes>`_ - Quantization library used in QLoRA
* `QLoRA Explained - Medium Article <https://medium.com/@dillipprasad60/qlora-explained-a-deep-dive-into-parametric-efficient-fine-tuning-in-large-language-models-llms-c1a4794b1766>`_ - Detailed tutorial and explanation
* `Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`_ - Training framework with QLoRA support used in FinLoRA