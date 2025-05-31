LoRA Foundations
============================

.. contents::
   :local:
   :depth: 4

What is LoRA?
-------------
LoRA is a technique to efficiently update the parameters of pre-trained language models when fine-tuning on new tasks.

Foundations of LoRA
-------------------
In this subsection, we introduce two fundamental concepts needed to understand LoRA—ranks and fine-tuning.

Ranks
~~~~~
Rank is the number of linearly independent rows or columns in a matrix.
Linearly independent columns, for example, are columns whose entries cannot be written as an integer-weighted sum of earlier columns.

.. math::

   W =
   \begin{bmatrix}
    1 & 7 & 2 & 8 & 5\\
    2 & 10 & 4 & 12 & 10\\
    3 & 15 & 12 & 18 & 27\\
    4 & 12 & 16 & 16 & 36
   \end{bmatrix},
   \qquad
   \text{Dimensions: }4 \times 5\;(\text{rows}\times\text{columns})

In this matrix there are **two** linearly independent columns, so
:math:`\operatorname{rank}(W)=2`.

* Column 1 is independent (nothing precedes it).
* Column 2 cannot be written as a multiple of Column 1, so it is also independent.
* Columns 3–5 are dependent:

.. math::

     C_3 = 2C_1,\qquad
     C_4 = C_1 + C_2,\qquad
     C_5 = C_1 + 2C_2.

Re-expressing those dependencies in vector form:

.. math::

   W \;=\;
   \underbrace{\begin{bmatrix}
    1 & 7\\
    2 & 10\\
    3 & 15\\
    4 & 12
   \end{bmatrix}}_{B\in\mathbb{R}^{4\times2}}
   \;
   \underbrace{\begin{bmatrix}
    1 & 0 & 2 & 1 & 1\\
    0 & 1 & 0 & 1 & 2
   \end{bmatrix}}_{A\in\mathbb{R}^{2\times5}}.

.. math::

   \begin{aligned}
   \text{Dimensions}(W)     &= d\times k = 4\times5,\\
   \text{Dimensions}(B)     &= d\times r = 4\times2,\\
   \text{Dimensions}(A)     &= r\times k = 2\times5,\\
   \text{Dimensions}(BA)    &= d\times k = \text{Dimensions}(W).
   \end{aligned}

.. math::

   \begin{aligned}
   \text{Parameters}(W) &= 4\times5 = 20,\\
   \text{Parameters}(B) &= 4\times2 = 8,\\
   \text{Parameters}(A) &= 2\times5 = 10,\\
   \text{Parameters}(BA)  &= 8 + 10 = 18.
   \end{aligned}

Thus storing :math:`B` and :math:`A` uses fewer parameters than storing :math:`W` directly—a key idea behind *low-rank* adaptation (LoRA).

Full Fine-Tuning
~~~~~~~~~~~~~~~~~

Consider a pre-trained model M with 500 million parameters. Suppose we pre-trained M with two tasks. Task 1 is Masked Language Modeling (MLM), where we mask some words in a sentence, and the task is to predict the sentence with the masked tokens filled in. Task 2 is Next Sentence Prediction (NSP), where the task is to predict if, given 2 sentences, whether or not sentence A comes before sentence B.

If we want to fine-tune the pre-trained model M on a new task, Named Entity Recognition (NER), where the task is to annotate one entity (location/person/organization) per sentence in a financial task.

When we perform full fine-tuning on model M, all parameters are updated based on the gradients we compute during backpropagation. In backpropagation, we compute the loss (the difference between the predicted output and the target output) and propagate the loss backward through the model. As we propagate the loss backward, we compute the gradient of the loss with respect to each parameter. The optimizer uses these gradients to update the model's parameters.

If we want to fine-tune model M on another task Financial Phrase Bank (FPB), where the task is to annotate sentences from financial news and reports with sentiment, we still need to update all 500 million parameters. This is costly and can lead to overfitting and the model forgetting pre-training tasks.

Fine-Tuning With Adapters (Parameter-Efficient Fine-Tuning—PEFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter Efficient Fine-Tuning (PEFT) adds small adapter layers per transformer block as shown below. Let's consider a scenario in which we use PEFT to fine-tune the pre-trained model M and add two adapter layers per transformer layer.

Now, when we fine-tune M on NER, only the adapter parameters are updated. This only consists of a tiny fraction of the original parameters. The rest of the model's parameters are frozen. This means that, during backpropagation, the gradients of loss pass through them, but the parameters aren't updated. While we do have to swap the adapters and store the updated parameters separately for FPB, the number of parameters required to fine-tune on FPB is much smaller than full fine-tuning.

LoRA Methods
------------
In this subsection, we introduce the five LoRA methods we use in our paper. We choose LoRA [1]_ and QLoRA [2]_ due to their standard use in fine-tuning. We choose DoRA [3]_ and rsLoRA [4]_ due to their performance enhancements: DoRA proposes fine-grained updates for achieving accuracy through LoRA, and rsLoRA proposes a scaling factor to achieve gradient stability. Lastly, we choose LoRA with federated learning for its practical ability to allow financial institutions to collaborate in fine-tuning models while using private, confidential data.

Low-Rank Adaptation (LoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LoRA adds a scaled low-rank update :math:`\Delta \boldsymbol{W} = \gamma_r\boldsymbol{B}\boldsymbol{A}`—where :math:`\gamma_r` is a scaling factor (:math:`\gamma_r=\frac{\alpha}{r}` with :math:`\alpha` > 0 and rank :math:`r` > 0), :math:`\boldsymbol{B} \in \mathbb{R}^{d \times r}`, and :math:`\boldsymbol{A} \in \mathbb{R}^{r \times k}`—to the frozen pre-trained weight matrix :math:`\boldsymbol{W}_0 \in \mathbb{R}^{d \times k}`.

For each multi-head attention layer, we have query, key, and value weight matrices, which we can factorize as follows:

.. math::

   W_Q^{(n)} = B_Q^{(n)}A_Q^{(n)},\quad
   W_K^{(n)} = B_K^{(n)}A_K^{(n)},\quad
   W_V^{(n)} = B_V^{(n)}A_V^{(n)}.

During fine-tuning, the weight matrices are updated as follows with the scaled low-rank update:

.. math::

   \begin{aligned}
   W_{Q,\text{new}}^{(n)} &= W_{Q,\text{old}}^{(n)} + \gamma_rB_Q^{(n)}A_Q^{(n)},\\
   W_{K,\text{new}}^{(n)} &= W_{K,\text{old}}^{(n)} + \gamma_rB_K^{(n)}A_K^{(n)},\\
   W_{V,\text{new}}^{(n)} &= W_{V,\text{old}}^{(n)} + \gamma_rB_V^{(n)}A_V^{(n)}.
   \end{aligned}

Because the update is in-place, no extra layers are added, and inference latency is unchanged.

References
----------

.. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.

.. [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.

.. [3] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024, July). Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

.. [4] Kalajdzievski, D. (2023). Rank-stabilized scaling factor for LoRA adaptation.

.. [5] Liu, X. Y., Zhu, R., Zha, D., Gao, J., Zhong, S., White, M., & Qiu, M. (2025). Differentially private low-rank adaptation of large language model using federated learning. ACM Transactions on Management Information Systems, 16(2), 1-24.