
Weight-Decomposed Low-Rank Adaptation (DoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LoRA makes simple changes to the model weights, so it sometimes doesn't capture the full complexity of the data and its relationships. DoRA solves this issue of capturing data complexity.

DoRA decomposes the weight matrix into a *magnitude vector* and a *direction matrix*.
The magnitude vector consists of the lengths of the columns in the weight matrix and is computed by taking each column's :math:`\ell_2` norm.
The direction matrix :math:`\boldsymbol V` is the collection of the original columns. Its unit-column form :math:`\widehat{\boldsymbol V}=\boldsymbol V/\lVert\boldsymbol V\rVert_c` is obtained by dividing each column by its :math:`\ell_2` norm.

The magnitude vector :math:`\boldsymbol{m}` is of size :math:`1 \times k`, where :math:`k` is the number of columns. The direction matrix :math:`\boldsymbol{V}` is of size :math:`d \times k`, where :math:`d` is the number of rows in a weight matrix.

The decomposition can be written as:

.. math::

   \boldsymbol{W}_0 \;=\; \boldsymbol{m}\,\frac{\boldsymbol{V}}{\lVert \boldsymbol{V} \rVert_c}\;=\;\lVert \boldsymbol{W}_0 \rVert_c\,\frac{\boldsymbol{W}_0}{\lVert \boldsymbol{W}_0 \rVert_c},

where :math:`\lVert \cdot \rVert_c` denotes the column-wise :math:`\ell_2` norm (i.e., the norm is taken independently for each column) and :math:`\boldsymbol{W}_0` is the frozen pretrained weight.

Here is an example of the decomposition:

.. math::

   \boldsymbol{W}_0 =
   \begin{bmatrix}
   1 & 7 & 2 & 8 & 5 \\
   2 & 10 & 4 & 12 & 10 \\
   3 & 15 & 12 & 18 & 27 \\
   4 & 12 & 16 & 16 & 36
   \end{bmatrix}, \qquad
   \boldsymbol{W}_0 \in \mathbb{R}^{4 \times 5}.

For column :math:`j`

.. math::

   \lVert \boldsymbol{w}_j \rVert_2 = \sqrt{\sum_{i=1}^{4} W_{ij}^2}.

These norms form a :math:`1 \times 5` magnitude vector:

.. math::

   \boldsymbol{m} = \left[ 5.4772,\; 22.7596,\; 20.4939,\; 28.0713,\; 46.3681 \right].

The unit-column direction matrix is

.. math::

   \widehat{\boldsymbol{V}} =
   \begin{bmatrix}
   0.182574 & 0.307562 & 0.097590 & 0.284988 & 0.107833 \\
   0.365148 & 0.439375 & 0.195180 & 0.427482 & 0.215666 \\
   0.547723 & 0.659062 & 0.585540 & 0.641223 & 0.582297 \\
   0.730297 & 0.527250 & 0.780720 & 0.569976 & 0.776396
   \end{bmatrix}.

Every column of :math:`\widehat{\boldsymbol{V}}` now has unit length:

.. math::

   \lVert \boldsymbol{v}_j \rVert_2 = 1, \qquad \text{for all } j.

These are updated separately. The magnitude vector :math:`\boldsymbol{m}` is trained directly, while the direction matrix :math:`\boldsymbol{V}` is fine-tuned using LoRA: :math:`\Delta\boldsymbol{V} = \boldsymbol{B}\boldsymbol{A}` with :math:`\boldsymbol{B}\!\in\!\mathbb{R}^{d\times r}` and :math:`\boldsymbol{A}\!\in\!\mathbb{R}^{r\times k}`.

After the updates, the new weight matrix is

.. math::

   \boldsymbol{W}' = \boldsymbol{m}\,\frac{\boldsymbol{V} + \Delta \boldsymbol{V}}{\lVert \boldsymbol{V} + \Delta \boldsymbol{V} \rVert_c}
        = \boldsymbol{m}\,\frac{\boldsymbol{W}_0 + \boldsymbol{B}\boldsymbol{A}}{\lVert \boldsymbol{W}_0 + \boldsymbol{B}\boldsymbol{A} \rVert_c}.

Using DoRA in FinLoRA
----------------------

To use DoRA in FinLoRA, you need to set the ``peft_use_dora`` parameter to ``true`` in your configuration. Here's an example of how to configure DoRA for fine-tuning:

.. code-block:: bash

   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8_dora

This uses the configuration from ``lora/finetune_configs.json``:

.. code-block:: json

   "sentiment_llama_3_1_8b_8bits_r8_dora": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 8,
     "quant_bits": 8,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2,
     "peft_use_dora": true
   }

You can also specify a custom ``lora_alpha`` value for DoRA:

.. code-block:: json

   "sentiment_llama_3_1_8b_8bits_r8_dora_a32": {
     "base_model": "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
     "lora_r": 8,
     "quant_bits": 8,
     "learning_rate": 0.0001,
     "num_epochs": 4,
     "batch_size": 8,
     "gradient_accumulation_steps": 2,
     "lora_alpha": 32,
     "peft_use_dora": true
   }

The key parameters for DoRA are:
- ``peft_use_dora``: Set to ``true`` to enable DoRA
- ``lora_r``: The rank of the LoRA adapter
- ``lora_alpha``: The scaling factor for the LoRA adapter (optional, default is 16)

DoRA adapters are saved in the ``lora_adapters/8bits_r8_dora`` directory after fine-tuning.
