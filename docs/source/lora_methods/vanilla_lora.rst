Vanilla LoRA
============================

.. contents:: Table of Contents

Background
----------

**Citation:** `LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) <https://arxiv.org/abs/2106.09685>`_

LoRA addresses the fundamental challenge of full fine-tuning large language models (LLMs), which is it becomes increasingly impractical as models grow larger due to requiring more compute and storage. LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, enabling efficient adaptation to downstream tasks. It can reduce the number of trainable parameters by up to 10,000 times and GPU memory requirements by three times.

Quick Facts
~~~~~~~~~~~

#. LoRA is a parameter-efficient fine-tuning method.
#. LoRA can reduce trainable parameters by up to 10,000x compared to full fine-tuning.
#. LoRA introduces no additional inference latency when weights are merged.

Algorithmic Idea
~~~~~~~~~~~~~~~~

The core idea behind LoRA is that the change in weights during model adaptation has a low "intrinsic rank". LoRA preserves the weights of the pre-trained model and introduces a smaller set of trainable weights through low-rank decomposition.

For a pre-trained weight matrix :math:`\mathbf{W}_0 \in \mathbb{R}^{d \times k}`, instead of updating all parameters, LoRA only updates a small subset of parameters using low-rank matrices :math:`\mathbf{A} \in \mathbb{R}^{r \times k}` and :math:`\mathbf{B} \in \mathbb{R}^{d \times r}` where the rank :math:`r \ll \min(d,k)`.

During training, the following hold true:

#. :math:`\mathbf{W}_0` is frozen and receives no gradient updates.
#. Only :math:`\mathbf{A}` and :math:`\mathbf{B}` contain trainable parameters with significantly fewer parameters.
#. The forward pass becomes: :math:`\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}`.
#. :math:`\mathbf{A}` is initialized randomly while :math:`\mathbf{B}` is initialized to zero.

Key Equations
~~~~~~~~~~~~

For a pre-trained weight matrix :math:`\mathbf{W}_0 \in \mathbb{R}^{d \times k}`, the LoRA update follows:

.. math::

   \mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta\mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}

Where the low-rank decomposition is:

.. math::

   \Delta\mathbf{W} = \gamma_r \mathbf{B}\mathbf{A}

The scaling factor is defined as:

.. math::

   \gamma_r = \frac{\alpha}{r}

Where:

#. :math:`\alpha > 0` is a hyperparameter that controls the scaling.
#. :math:`r > 0` is the rank with the low-rank condition :math:`r \ll \min(d,k)`.
#. :math:`\mathbf{A} \in \mathbb{R}^{r \times k}` is initialized with random Gaussian weights.
#. :math:`\mathbf{B} \in \mathbb{R}^{d \times r}` is initialized to zero, so :math:`\Delta\mathbf{W} = 0` at training start.

The number of trainable parameters is:

.. math::

   |\Theta| = 2 \times L_{\text{LoRA}} \times d_{\text{model}} \times r

where :math:`L_{\text{LoRA}}` is the number of weight matrices LoRA is applied to, :math:`d_{\text{model}}` is the model dimension, and :math:`r` is the rank.

Implementation in FinLoRA
~~~~~~~~~~~~~~~~~~~~~~~~

To use vanilla LoRA in FinLoRA, configure fine-tuning with standard parameters:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8

Configuration example from ``lora/finetune_configs.json``:

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

Key parameters:
- ``lora_r``: The rank :math:`r` of the LoRA adapter (typically 4-16)
- ``quant_bits``: The quantization bits (we use 8 for vanilla LoRA, but different numbers of quant bits can be used)
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

   # Load LoRA adapter
   adapter_path = "./lora_adapters/8bits_r8/sentiment_llama_3_1_8b_8bits_r8"
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

.. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.

Why This Paper?
~~~~~~~~~~~~~~~

The original LoRA paper is important to understanding parameter-efficient fine-tuning. It introduces the core mathematical techniques that all subsequent LoRA variants build upon. The paper provides theoretical justification for low-rank adaptations and has been widely adopted for fine-tuning LLMs.

Useful Links
~~~~~~~~~~~~

* `Microsoft LoRA <https://github.com/microsoft/LoRA>`_ - Original implementation by the authors
* `LoRA Explained by Primary Author <https://www.youtube.com/watch?v=DhRoTONcyZE>`_ - Production-ready LoRA implementation
* `Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`_ - Training framework with LoRA support used in FinLoRA 