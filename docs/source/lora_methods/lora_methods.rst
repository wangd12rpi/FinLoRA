LoRA Foundations and Methods
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

Quantized LoRA (QLoRA)
~~~~~~~~~~~~~~~~~~~~~~

When fine-tuning, LoRA requires a large amount of GPU memory. To solve this issue, we can use QLoRA. 
QLoRA drastically reduces memory usage and lets you fine-tune on a single GPU.

In QLoRA, we quantize the weights of the adapter layers, reducing both parameter count and memory usage. Quantization is a technique that reduces the precision of the weights to reduce the number of bits used to store them. It consists of two parts: Rounding to the nearest integer and truncating to remove the decimal portion of a floating point number. QLoRA specifically uses 4-bit NormalFloat (NF4), an optimal data type for normally distributed weights, quantization. Pre-trained weights are usually normally distributed and centered around 0, which is why NF4 is ideal for quantization.

If we quantize from Float16 to Int4, we can represent 16 different values (bins) because Int4 has 4 bits and :math:`2^{4}=16`. Inputs are usually normalized from -1 to 1. Very close together values, however, will be mapped to the same bin. This means that the precision is lost if we want to convert back to Float16. However, we can use blockwise quantization, where we divide the input range into blocks and quantize each block separately. QLoRA uses a 64 blocksize for better precision.

Since regular quantization relies on the bins being equally probable, QLoRA uses NormalFloat where the bins are weighted by the normal distribution. The spacing between bins is therefore closer together near 0 and further apart further away from 0.

Each block in QLoRA has a quantization constant. QLoRA employs double quantization, where it quantizes the quantization constants themselves to further save space.

The last part of QLoRA is paged optimizers. Paged optimizers reduce GPU memory spikes by switching pages to CPU memory when GPU RAM becomes full when processing long sequences, and the pages are not needed for the current computation of the forward/backward pass.

The forward pass for QLoRA is :math:`\boldsymbol{y} = p_{16}(\boldsymbol{W}_0^{\text{NF4}}) \boldsymbol{x} + \gamma_r \boldsymbol{B} \boldsymbol{A} \boldsymbol{x}`.

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

Rank-Stabilized LoRA (rsLoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LoRA scales the weight matrix update :math:`\boldsymbol{BA}` by :math:`\frac{\alpha}{r}`, which can cause gradients to explode or diminish as the rank :math:`r` increases. In contrast, rsLoRA uses a scaling factor :math:`\frac{\alpha}{\sqrt{r}}`:

.. math::

   \boldsymbol W'=\boldsymbol W_0+\frac{\alpha}{\sqrt{r}}\boldsymbol B\boldsymbol A.

This scaling results in gradient-scale stability at higher ranks, enabling the rank to be higher to capture more details in long-context tasks like XBRL extraction. rsLoRA also results in lower perplexity—the model assigns higher probabilities to correct words—than LoRA at higher ranks.

LoRA with Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the finance sector, multiple banks may want to work together on a model to predict credit risk and whether a borrower will default on a loan. Each bank may have a different dataset, but they cannot share their data due to compliance reasons and privacy concerns. Federated learning solves this issue by fine-tuning a model on local data and aggregating updates during backpropagation to a centralized model via a server.

Differentially Private Low-Rank Adaptation (DP-LoRA) [5]_ is a method to use federated learning with LoRA.

DP-LoRA first uses a server to send the current global LoRA weights (the A and B matrices from earlier) to all clients.

Every client does the following: 1) Gets a minibatch of its private data 2) Computes the gradient for only its local A and B weights clipped with an :math:`\ell_2` norm (square root of the sum of the squares of elements in the vector) 3) Adds Gaussian noise to the gradients 4) Updates the A and B LoRA matrices 5) Sends the updated A and B matrices to the server.

By adding noise, DP-LoRA prevents the centralized model from inferring the private data later on. This would allow the banks in the credit risk example to work on a model together.

As in normal federated learning, the server then aggregates the weights from all clients in a weighted average and sends the updated weights to all clients.

References
----------

.. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.

.. [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.

.. [3] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024, July). Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

.. [4] Kalajdzievski, D. (2023). Rank-stabilized scaling factor for LoRA adaptation.

.. [5] Liu, X. Y., Zhu, R., Zha, D., Gao, J., Zhong, S., White, M., & Qiu, M. (2025). Differentially private low-rank adaptation of large language model using federated learning. ACM Transactions on Management Information Systems, 16(2), 1-24.