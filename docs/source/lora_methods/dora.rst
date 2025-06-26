DoRA
====

.. contents:: Table of Contents

Background
----------

**Citation:** `DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024) <https://arxiv.org/abs/2402.09353>`_

DoRA introduces improvements that are intended to close the issue of LoRA's accuracy lagging behind that of full fine-tuning. DoRA decomposes pre-trained weights into magnitude and direction components, fine-tuning both while using LoRA specifically for directional updates to efficiently minimize trainable parameters. It enhances both learning capacity and training stability while avoiding additional inference overhead.

Quick Facts
~~~~~~~~~~~

#. DoRA uses weight-decomposed fine-tuning to extend LoRA with magnitude-direction decomposition.
#. DoRA can often but not in all cases (such as ours) achieve accuracy close to full fine-tuning while maintaining the same parameter count as LoRA.
#. DoRA introduces no additional inference latency when weights are merged.

Algorithmic Idea
~~~~~~~~~~~~~~~~

LoRA's limitations come from coupling magnitude and direction updates. DoRA separates those components, enabling mo43 fine-grained adaptation that more closely matches full fine-tuning.

For a pre-trained weight matrix :math:`\mathbf{W}_0`, DoRA decomposes it into a magnitude vector :math:`\mathbf{m}` and direction matrix :math:`\mathbf{V}` where :math:`\mathbf{m} = ||\mathbf{W}_0||_c` (column-wise norm) and :math:`\mathbf{V} = \mathbf{W}_0`. The magnitude vector consists of the :math:`\ell_2` norms of each column, while the direction matrix contains the original weight matrix.

During fine-tuning, the following hold true:

#. :math:`\mathbf{W}_0` is decomposed into a magnitude component :math:`\mathbf{m}` and a direction component :math:`\mathbf{V}`.
#. Only the direction matrix receives LoRA updates :math:`\Delta\mathbf{V} = \mathbf{B}\mathbf{A}` while the magnitude vector is trained directly.
#. The forward pass becomes: :math:`\mathbf{W}' = \mathbf{m} \frac{\mathbf{V} + \Delta\mathbf{V}}{||\mathbf{V} + \Delta\mathbf{V}||_c}`.

Key Equations
~~~~~~~~~~~~

For a pre-trained weight matrix :math:`\mathbf{W}_0 \in \mathbb{R}^{d \times k}`, the DoRA decomposition follows:

.. math::

   \mathbf{W}_0 = \mathbf{m} \frac{\mathbf{V}}{||\mathbf{V}||_c} = ||\mathbf{W}_0||_c \frac{\mathbf{W}_0}{||\mathbf{W}_0||_c}

Where the weight decomposition is:

.. math::

   \mathbf{m} = ||\mathbf{W}_0||_c, \quad \mathbf{V} = \mathbf{W}_0

The updated weight matrix becomes:

.. math::

   \mathbf{W}' = \mathbf{m} \frac{\mathbf{V} + \Delta\mathbf{V}}{||\mathbf{V} + \Delta\mathbf{V}||_c} = \mathbf{m} \frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{||\mathbf{W}_0 + \mathbf{B}\mathbf{A}||_c}

Where:

#. :math:`\mathbf{m} \in \mathbb{R}^{1 \times k}` is the magnitude vector of column-wise :math:`\ell_2` norms.
#. :math:`\mathbf{V} \in \mathbb{R}^{d \times k}` is the direction matrix initialized as :math:`\mathbf{W}_0`.
#. :math:`\mathbf{A} \in \mathbb{R}^{r \times k}` and :math:`\mathbf{B} \in \mathbb{R}^{d \times r}` are the LoRA adaptation matrices for directional updates.
#. :math:`||\cdot||_c` denotes the column-wise :math:`\ell_2` norm operation.

The number of trainable parameters is:

.. math::

   |\Theta| = k + 2 \times L_{\text{DoRA}} \times d_{\text{model}} \times r

where :math:`k` accounts for the magnitude vector, :math:`L_{\text{DoRA}}` is the number of weight matrices DoRA is applied to, :math:`d_{\text{model}}` is the model dimension, and :math:`r` is the rank.

Implementation in FinLoRA
~~~~~~~~~~~~~~~~~~~~~~~~

To use DoRA in FinLoRA, configure fine-tuning with DoRA enabled:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8_dora

Configuration example from ``lora/finetune_configs.json``:

.. code-block:: json

   "sentiment_llama_3_1_8b_8bits_r8_dora": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 8,
     "quant_bits": 8,
     "peft_use_dora": true,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2
   }

Key parameters:
- ``lora_r``: The rank :math:`r` of the LoRA adapter (typically 8-16 for DoRA)
- ``quant_bits``: The quantization bits (8 or 4, same as vanilla LoRA)
- ``peft_use_dora``: Enable DoRA decomposition (set to true)
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

   # Load DoRA adapter
   adapter_path = "./lora_adapters/8bits_r8_dora/sentiment_llama_3_1_8b_8bits_r8_dora"
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

.. [1] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *arXiv preprint arXiv:2402.09353*.

Why This Method?
~~~~~~~~~~~~~~~

DoRA is important because it addresses a funadmental flaw in LoRA that causes it to lag behind in accuracy compared to full fine-tuning. It shows how decomposing into magnitude and direction components can enhance fine-tuning by allowing the model to capture more fine-grained patterns.

Useful Links
~~~~~~~~~~~~

* `NVIDIA DoRA Implementation <https://github.com/NVlabs/DoRA>`_ - Official implementation by NVIDIA
* `NVIDIA Technical Blog: Introducing DoRA <https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/>`_ - Technical blog about DoRA by NVIDIA
* `Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`_ - Training framework with DoRA support used in FinLoRA
