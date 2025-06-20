==================
Overview
==================

Introduction
============

Low-rank adaptation (LoRA) methods show great potential for scaling pre-trained general-purpose
Large Language Models (LLMs) to hundreds or thousands of use scenarios. However, their efficacy in
high-stakes domains like finance is rarely explored, e.g., passing CFA exams and analyzing SEC
filings. In this paper, we present the open-source FinLoRA project that benchmarks LoRA methods on
both general and highly professional financial tasks. First, we curated 19 datasets covering diverse
financial applications; in particular, we created four novel XBRL analysis datasets based on 150 SEC
filings. Second, we evaluated five LoRA methods and five base LLMs. Finally, we provide extensive
experimental results in terms of accuracy, F1, and BERTScore and report computational cost in terms
of time and GPU memory during fine-tuning and inference stages. We find that LoRA methods achieved
substantial performance gains of 36 % on average over base models. Our FinLoRA project provides an
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

Performance of FinLoRA
======================

.. raw:: html

    <object class="figure" data="../_static/images/p1_new.svg" type="image/svg+xml"></object>
    <br>

References
==========

.. _BloombergGPT: https://arxiv.org/abs/2303.17564
