General Financial Tasks
=======================

We consider six general financial tasks, in total 122.9k train questions and 31.7k test questions.

.. list-table:: Overview of general financial tasks.
   :widths: auto
   :header-rows: 1

   * - Question sets
     - Type
     - #Train
     - #Test
     - Average Prompt Length
     - Metrics
     - Source
     - Train Data
     - Test Data
   * - Financial Phrase Bank (FPB)
     - Sentiment Analysis
     - 3.1k
     - 970
     - 56
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/TheFinAI/en-fpb>`__
     - Part of `finlora_sentiment_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/finlora_sentiment_train.jsonl>`__
     - `fpb_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/fpb_test.jsonl>`__
   * - Financial Question Answering (FiQA SA)
     - Sentiment Analysis
     - 822
     - 234
     - 48
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification>`__
     - Part of `finlora_sentiment_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/finlora_sentiment_train.jsonl>`__
     - `fiqa_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/fiqa_test.jsonl>`__
   * - Twitter Financial News Sentiment (TFNS)
     - Sentiment Analysis
     - 9.5k
     - 2.4k
     - 52
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment>`__
     - Part of `finlora_sentiment_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/finlora_sentiment_train.jsonl>`__
     - `tfns_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/tfns_test.jsonl>`__
   * - News with GPT (NWGI)
     - Sentiment Analysis
     - 12.9k
     - 4.1k
     - 81
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/TheFinAI/NWGI_test>`__
     - Part of `finlora_sentiment_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/finlora_sentiment_train.jsonl>`__
     - `nwgi_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/nwgi_test.jsonl>`__
   * - Headline
     - Headline Analysis
     - 82.2k
     - 20.5k
     - 43
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/FinGPT/fingpt-headline-cls>`__
     - `headline_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/headline_train.jsonl>`__
     - `headline_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/headline_test.jsonl>`__
   * - Named Entity Recognition (NER)
     - Named Entity Recognition
     - 13.5k
     - 3.5k
     - 138
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/FinGPT/fingpt-ner-cls>`__
     - `ner_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/ner_train.jsonl>`__
     - `ner_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/ner_test.jsonl>`__



**Financial Phrase Bank (FPB)** (Sentiment Analysis) [fpb]_
--------------------
Financial Phrase Bank (FPB) contains sentences extracted from financial news and reports. These sentences are annotated with sentiment labels "positive", "negative", and "neutral". We manually created the train/test split.


.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
   * - **Input**
     - Pharmaceuticals group Orion Corp reported a fall in its third-quarter earnings that were hit by larger expenditures on R&D and marketing.
   * - **Output**
     - negative

**FiQA SA** (Sentiment Analysis) [fiqa]_
--------------------

Financial question-answering sentiment analysis (FiQA SA) is another sentiment analysis dataset with the same labels as FPB from microblog headlines and financial news.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Instruction**
     - What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
   * - **Input**
     - Johnson Matthey raises prospect of investor payout
   * - **Output**
     - positive

**Twitter Financial News Sentiment (TFNS)** (Sentiment Analysis) [tfns]_
--------------------
Twitter financial news sentiment (TFNS) comprises annotated tweets related to financial news labeled with the same sentiment categories as FPB.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Instruction**
     - What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.
   * - **Input**
     - $BYND - JPMorgan reels in expectations on Beyond Meat https://t.co/bd0xbFGjkT
   * - **Output**
     - negative

**News with GPT (NWGI)** (Sentiment Analysis)
--------------------
News with GPT instruction (NWGI) comprises samples with seven labels ranging from strong negative to strong positive.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Instruction**
     - What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}.
   * - **Input**
     - Amid a soft performance for the major equity indices on Tuesday, Nvidia (NASDAQ: NVDA ) posted a particularly glaring loss. Shares continued to fall in sympathy with fellow semiconductor specialist Micron Technology (NASDAQ: MU ) following its disappointing earnings results last week.
   * - **Output**
     - moderately negative

**Financial Headline Analysis** (Headline Analysis) [headline-tasks]_
--------------------
The Headline dataset classifies headlines based on various questions into two classes: "yes" and
"no".

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Instruction**
     - Does the news headline talk about price? Please choose an answer from {Yes/No}.
   * - **Input**
     - Gold futures edge up after two-session decline
   * - **Output**
     - No

**Named Entity Recognition (NER)** (Named Entity Recognition) [ner-tasks]_
--------------------

The NER dataset annotates one entity per sentence, categorized into one of three classes: "location", "person", and "organization".

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Instruction**
     - What is the entity type of '40 William St' in the input sentence. Options: person, location, organization
   * - **Input**
     - This LOAN AND SECURITY AGREEMENT dated January 27 , 1999 , between SILICON VALLEY BANK (" Bank "), a California - chartered bank with its principal place of business at 3003 Tasman Drive , Santa Clara , California 95054 with a loan production office located at 40 William St ., Ste .
   * - **Output**
     - location


Fine-tuning for General Financial Tasks
--------------------------------------------------

To fine-tune a model for general financial tasks, you can use the configurations provided in the ``lora/finetune_configs.json`` file. Below are the configurations for each task:

Sentiment Analysis
^^^^^^^^^^^^^^^^^^^^^

To fine-tune a model for sentiment analysis tasks (FPB, FiQA SA, TFNS, NWGI), you can use one of the following configurations:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py sentiment_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py sentiment_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **sentiment_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **sentiment_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **sentiment_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **sentiment_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.

Headline Analysis
^^^^^^^^^^^^^^^^^^^^^

To fine-tune a model for the Headline Analysis task, you can use one of the following configurations:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py headline_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py headline_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py headline_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py headline_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **headline_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **headline_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **headline_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **headline_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.

Named Entity Recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To fine-tune a model for the Named Entity Recognition task, you can use one of the following configurations:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py ner_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py ner_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py ner_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py ner_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **ner_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **ner_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **ner_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **ner_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.

Citations
****************
.. [fpb] Malo, P., H. Lu, M. Ahlgren, S. Rönnqvist, and P. Nyberg. (2014). *FinancialPhraseBank-v1.0*. Available at SSRN: https://ssrn.com/abstract=2512146 or http://dx.doi.org/10.2139/ssrn.2512146
.. [fiqa] Sinha, A., Joglekar, M., & Murphy, F. (2018). *FiQA: Financial Opinion Mining and Question Answering*. arXiv preprint arXiv:1809.09431.
.. [tfns] Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv preprint arXiv:1908.10063.
.. [headline-tasks] Sinha, A., & Khandait, P. (2020). *Headline-Enhanced Financial Embedding*. In Proceedings of the 2nd Workshop on Economics and Natural Language Processing (pp. 66-74).
.. [ner-tasks] Salinas Alvarado, D., Rönnqvist, S., & Niklaus, J. (2015). *Domain-Specific Named Entity Recognition: A Case Study in Finance*. In Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing (pp. 110-115).
