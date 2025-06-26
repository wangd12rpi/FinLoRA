================
rsLoRA
================

.. contents:: Table of Contents

Background
----------

**Citation:** `A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (Kalajdzievski, 2023) <https://arxiv.org/abs/2312.03732>`_

rsLoRA addresses a limitation of vanilla LoRA where the scaling factor :math:`\alpha/r` can cause gradient instability as rank increases, which means fine-tuning can be unstable at high ranks in practice. rsLoRA introduces fixes this issue by using a rank-stabilized scaling factor :math:`\alpha/\sqrt{r}` that maintains gradient stability at higher ranks, enabling higher-rank adapters to be used for increased performance on complex tasks without additional inference cost.

Quick Facts
~~~~~~~~~~~

#. rsLoRA only changes the scaling factor that's used in LoRA, changing it from :math:`\alpha/r` to :math:`\alpha/\sqrt{r}`.
#. rsLoRA enables stable fine-tuning at higher ranks, which can increase performance for complex tasks.
#. rsLoRA has the same inference cost as LoRA.

Algorithmic Idea
~~~~~~~~~~~~~~~~

The core insight behind rsLoRA is that vanilla LoRA's scaling factor can cause the gradient magnitude to collapse as rank increases. This can prevent effective fine-tuning at higher ranks. rsLoRA's rank-stabilized scaling factor enables higher-rank adapters to be used for increased performance on complex tasks without additional inference cost.

During fine-tuning, the following hold true:

#. The :math:`\alpha/\sqrt{r}` scaling factor in rsLoRA leads to consistent gradient magnitudes across all ranks, allowing rsLoRA to capture more complex details for nuanced and complex tasks.

Key Equations
~~~~~~~~~~~~~

The rsLoRA adapter modifies the pre-trained weight matrix :math:`\mathbf{W}_0 \in \mathbb{R}^{d \times k}` as:

.. math::
   
   \mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{\sqrt{r}} \mathbf{B} \mathbf{A}

where :math:`\mathbf{A} \in \mathbb{R}^{r \times d}` and :math:`\mathbf{B} \in \mathbb{R}^{k \times r}` with :math:`r \ll \min(d,k)`.

The key theoretical result proves that for rank-stabilized adapters, the scaling factor must satisfy:

.. math::
   
   \gamma_r \in \Theta\left(\frac{1}{\sqrt{r}}\right)

This ensures that both forward activations and backward gradients maintain :math:`\Theta(1)` magnitude regardless of rank :math:`r`.

Implementation in FinLoRA
~~~~~~~~~~~~~~~~~~~~~~~~~

FinLoRA automatically uses rsLoRA when the ``use_rsLoRA`` parameter is enabled in the configuration.

**Key Parameters:**

* :math:`r` (``lora_rank``): Adapter rank, can be set higher than vanilla LoRA for better performance
* :math:`\alpha` (``lora_alpha``): Scaling parameter, typically set to 16 or 32
* ``use_rsLoRA``: Boolean flag to enable rank-stabilized scaling

Usage Example
~~~~~~~~~~~~~

Enable rsLoRA in your FinLoRA configuration:

.. code-block:: yaml

   # Enable rsLoRA with higher rank for complex tasks
   use_rsLoRA: true
   lora_rank: 64        # Higher ranks work effectively with rsLoRA
   lora_alpha: 16
   quant_bits: 8
   
   # rsLoRA particularly beneficial for complex financial tasks
   dataset: "xbrl_extract_train.jsonl"
   model_name: "meta-llama/Llama-3.1-8B-Instruct"

The scaling factor :math:`\gamma_r = \alpha/\sqrt{r} = 16/\sqrt{64} = 2.0` enables gradient stability at higher ranks.

References
----------

.. [1] Kalajdzievski, D. (2023). A rank stabilization scaling factor for fine-tuning with lora. *arXiv preprint arXiv:2312.03732*.

Why This Method?
~~~~~~~~~~~~~~~

rsLoRA enables gradient stability at higher ranks, which allows researchers to achieve consistently better performance at higher ranks for complex tasks without additional inference cost. It is particularly valuable for complex financial NLP tasks where higher model capacity can capture nuanced and complex domain-specific patterns.

Useful Links
~~~~~~~~~~~~

* `rsLoRA Technical Blog by Author <https://huggingface.co/blog/damjan-k/rsLoRA>`_ - Technical blog by the rsLoRA paper's author
* `Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`_ - Training framework with rsLoRA support used in FinLoRA
