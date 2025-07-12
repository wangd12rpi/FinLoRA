Vanilla LoRA
============================

.. contents:: Table of Contents

Background
----------

**Citation:** `LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) <https://arxiv.org/abs/2106.09685>`_

LoRA addresses the fundamental challenge of fully fine-tuning large language models (LLMs), which becomes increasingly impractical as models grow larger. LoRA freezes pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture, enabling efficient adaptation to downstream tasks.

Quick Facts
~~~~~~~~~~~

#. LoRA is a parameter-efficient fine-tuning method.  
#. LoRA can reduce trainable parameters by ~6,340× when applied to the 70B rank-4 configuration.  
#. LoRA introduces no additional inference latency when weights are merged.


Algorithmic Idea
~~~~~~~~~~~~~~~~


The core idea behind LoRA is that the change in weights during model adaptation has a low "intrinsic rank." LoRA preserves the weights of the pre-trained model and introduces a smaller set of trainable weights through low-rank decomposition.

**Stage One: Fine-tuning Process**


1. **Add a second path**: Introduce two low-rank matrices :math:`\mathbf{A} \in \mathbb{R}^{r \times n}` and :math:`\mathbf{B} \in \mathbb{R}^{n \times r}` where the rank :math:`r \ll n`.

2. **Feedforward pass**: The forward pass becomes :math:`\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}`, where the contribution from the frozen weights is :math:`\mathbf{W}_0 \mathbf{x}` and the adapter contribution is :math:`\gamma_r \mathbf{B}\mathbf{A} \mathbf{x}`. This combined output is then used to compute the loss function.

3. **Backpropagation**: :math:`\mathbf{W}_0` is frozen and receives no gradient updates. Only :math:`\mathbf{A}` and :math:`\mathbf{B}` receive gradients and are updated during training. :math:`\mathbf{A}` is initialized randomly while :math:`\mathbf{B}` is initialized to zero, ensuring :math:`\Delta\mathbf{W} = 0` at training start.

**Stage Two: Inference Weight Merging**

After training, the learned adapter can be merged with the original weights for efficient inference:

.. math::

   \mathbf{W}_{merged} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \gamma_r \mathbf{B}\mathbf{A}, where \gamma_r = \frac{\alpha}{r}.

Once merged, inference becomes a standard matrix multiplication :math:`\mathbf{h} = \mathbf{W}_{merged} \mathbf{x}` with no additional computational overhead.

Only the small matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` require gradient computation and storage as LoRA adapters, dramatically reducing memory requirements and trainable parameters compared to full fine-tuning.


**Detailed Parameter Reduction with Rank=4 (Llama 3.1 70B)**

- **Full model parameters**: ~70.55B
- **LoRA applied to**: q_proj, k_proj, v_proj
  - **q_proj** (`in_features=8192`, `out_features=8192`):  
    :math:`(8192 \times 4) + (4 \times 8192) = 65{,}536`
  - **k_proj** (`in_features=8192`, `out_features=1024`):  
    :math:`(8192 \times 4) + (4 \times 1024) = 32{,}768 + 4{,}096 = 36{,}864`
  - **v_proj** (`in_features=8192`, `out_features=1024`):  
    :math:`(8192 \times 4) + (4 \times 1024) = 32{,}768 + 4{,}096 = 36{,}864`

- **Single attention block**: 65,536 + 36,864 + 36,864 = 139,264 LoRA parameters
- **For 80 blocks**: 139,264 × 80 = 11,141,120 total LoRA parameters
- **Reduction factor**:

  .. math::

     \frac{\text{Full model parameters}}{\text{LoRA parameters}}
     = \frac{70{,}553{,}706{,}496}{11{,}141{,}120}
     \approx 6{,}337

Thus, LoRA rank-4 adapts a 70B model using only ~1/6,337 of the total parameters, making large-model fine-tuning both memory-efficient and feasible.


Key Equations
~~~~~~~~~~~~

For a pre-trained weight matrix :math:`\mathbf{W}_0 \in \mathbb{R}^{n \times n}`, the LoRA update follows: :math:`\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta\mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \gamma_r \mathbf{B}\mathbf{A} \mathbf{x}`.

Where the low-rank decomposition is: :math:`\Delta\mathbf{W} = \gamma_r \mathbf{B}\mathbf{A}`.

The scaling factor is defined as: :math:`\gamma_r = \frac{\alpha}{r}`.

Where:

#. :math:`\alpha > 0` is a hyperparameter controlling the scaling.
#. :math:`r > 0` is the rank with the low-rank condition :math:`r \ll n`.
#. :math:`\mathbf{A} \in \mathbb{R}^{r \times n}` is initialized with random Gaussian weights.
#. :math:`\mathbf{B} \in \mathbb{R}^{n \times r}` is initialized to zero, so :math:`\Delta\mathbf{W} = 0` at training start.

The number of trainable parameters is: :math:`|\Theta| = 2 \times L_{\text{LoRA}} \times n \times r`.

where :math:`L_{\text{LoRA}}` is the number of weight matrices LoRA is applied to, :math:`n` is the matrix dimension, and :math:`r` is the rank.

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
- ``quant_bits``: The quantization bits (we use 8 for vanilla LoRA, but different values can be used)
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

Why This Method?
~~~~~~~~~~~~~~~

LoRA is crucial to understanding parameter-efficient fine-tuning. It introduced the core mathematical concepts upon which subsequent LoRA variants were based, providing theoretical justification for low-rank adaptations and widespread adoption for LLM fine-tuning.

Useful Links
~~~~~~~~~~~~


* `Microsoft LoRA <https://github.com/microsoft/LoRA>`_ - Original implementation  
* `LoRA Explained by Primary Author <https://www.youtube.com/watch?v=DhRoTONcyZE>`_  
* `Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`_ - Training framework with LoRA support used in FinLoRA
```
