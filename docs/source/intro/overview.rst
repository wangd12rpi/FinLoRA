==================
Overview
==================


Low-rank adaptation (LoRA) methods show great potential for scaling pre-trained general-purpose
Large Language Models (LLMs) to hundreds or thousands of use scenarios. However, their efficacy in
high-stakes domains like finance is rarely explored, e.g., passing CFA exams and analyzing SEC
filings. In this paper, we present the open-source FinLoRA project that benchmarks LoRA methods on
both general and highly professional financial tasks. First, we curated 19 datasets covering diverse
financial applications; in particular, we created four novel XBRL analysis datasets based on 150 SEC
filings. Second, we evaluated five LoRA methods and five base LLMs. Finally, we provide extensive
experimental results in terms of accuracy, F1, and BERTScore and report computational cost in terms
of time and GPU memory during fine-tuning and inference stages. We find that LoRA methods achieved
substantial performance gains of 36% on average over base models. Our FinLoRA project provides an
affordable and scalable approach to democratize financial intelligence to the general public.

Motivation
==========

The proprietary `BloombergGPT`_ model announced in April 2023 highlighted the potential of financial
Large Language Models (FinLLMs). However, such a “train-from-scratch” approach was resource-intensive,
requiring one million GPU hours at an estimated cost of \$3 million (\$3 per GPU hour in 2023) and
512 A100 GPUs. This substantial investment underscores the need for a cost-effective solution.

We propose to leverage open-source models, such as Llama 3.1, and employ the LoRA (Low-Rank Adaptation)
fine-tuning method. It dramatically reduces the number of trainable parameters to as little as 0.01 %
of the full model's parameters. This enables fine-tuning on 4 A5000 GPUs and brings the cost of
fine-tuning down to less than \$100, making FinLLMs accessible to the general public.

Performance
======================

.. raw:: html

    <object class="figure" data="../_static/images/p1_new.svg" type="image/svg+xml"></object>
    <br>

As illustrated in the performance comparison above, Llama 3.1 8B Intruct with our LoRA adpaters demonstrates substantial improvements across all financial task categories. The fine-tuned Llama 3.1 8B model using various LoRA methods achieves remarkable performance gains, with improvements ranging from +36.4% to +67.1% across different task types. Most notably, LoRA methods show exceptional effectiveness in **Financial Certificate** tasks (professional exams like CFA and CPA), where models achieve over 80% accuracy compared to the base model's 13-32% range. Similarly, our LoRA adpaters show significant improvements of +40% to +52% in **Financial Statement Analysis** tasks, particularly in our novel XBRL analysis datasets, highlighting LoRA's capability in handling complex, structured financial data.

The results reveal that, while larger base models like GPT-4o and DeepSeek V3 perform well on general financial tasks, our cost-effective LoRA-adapted Llama 3.1 8B models often match or exceed their performance while requiring only a fraction of the computational resources. This validates our approach of democratizing financial intelligence through parameter-efficient fine-tuning, making sophisticated financial AI accessible to organizations without massive computational budgets.


.. _BloombergGPT: https://arxiv.org/abs/2303.17564
