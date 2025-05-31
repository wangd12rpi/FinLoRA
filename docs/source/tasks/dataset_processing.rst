==================================================================
Dateset Processing
==================================================================


Training Data Preparation
-------------------------------

Sentiment Analysis
~~~~~~~~~~~~~~~~

The training data for the Sentiment Analysis LoRA model was constructed by aggregating and processing 4 distinct financial sentiment datasets.


1.  **Datasets Used:**

    *   Financial PhraseBank (FPB) ``financial_phrasebank``
    *   FiQA Sentiment Analysis (FiQA SA) ``ChanceFocus/flare-fiqasa``
    *   Twitter Financial News Sentiment (TFNS) ``zeroshot/twitter-financial-news-sentiment``
    *   News with GPT Instructions (NWGI) ``oliverwang15/news_with_gpt_instructions``

2.  **Common Processing Steps (Applied Before/During Splitting as appropriate):**

    *   **Label Normalization:** Labels were standardized. Numerical labels (FPB, TFNS) or sentiment scores (FiQA) were mapped to string labels (e.g., ``"negative"``, ``"neutral"``, ``"positive"``). FiQA scores were binned (-0.1, 0.1 thresholds). NWGI labels were kept as multi-class strings.
    *   **Instruction Formatting:** A specific ``instruction`` column was added.
    *   **Column Standardization:** Datasets were standardized to have ``input``, ``output``, and ``instruction`` columns.

3.  **Dataset-Specific Contribution and Splitting:**

    *   **Financial PhraseBank (FPB):**

        *   **Source:** Original ``train`` split (``sentences_50agree`` configuration).
        *   **Splitting:** Manual split (25% test, 75% train, seed 42)
        *   **Contribution to Combined Training Set:** The training portion from the split (duplicated 6 times)
        *   **Test Set:** The test portion from the split.

    *   **FiQA Sentiment Analysis (FiQA SA):**

        *   **Source:** The original ``train``, ``validation``, and ``test`` splits were loaded.
        *   **Splitting:** Original.
        *   **Contribution to Combined Training Set:** The original ``train`` split. (duplicated 21 times)
        *   **Test Set:** The original ``test`` split.

    *   **Twitter Financial News Sentiment (TFNS):**

        *   **Source:** The original ``train`` and ``test`` split was loaded and processed.
        *   **Splitting:** Original.
        *   **Contribution to Combined Training Set:** The original ``train`` split. (duplicated 2 times)
        *   **Test Set:** The original ``test`` split.

    *   **News with GPT Instructions (NWGI):**

        *   **Source:** The original ``train`` and ``test`` split was loaded and processed.
        *   **Splitting:** Original.
        *   **Contribution to Combined Training Set:** The original ``train`` split.
        *   **Test Set:** The original ``test`` split.

4.  **Final Combined Training Set Construction:**
    The processed and augmented **training portions** from FPB, FiQA, TFNS, and NWGI (as described above) were concatenated into a single large dataset and shuffled (seed=42). Total size PENDING.

5.  **Evaluation Strategy:**
    The fine-tuned Sentiment Analysis model was evaluated separately against each of the **test sets** created for FPB, FiQA, TFNS, and NWGI.


Headline Analysis
~~~~~~~~~~~~~~~~

The data preparation for the Headline Analysis LoRA model was more straightforward:

1.  **Dataset Used:** The standard Financial Headline Analysis dataset [headline]_ was used.
2.  **Train/Test Split:** The original ``train`` split provided with the dataset was used directly as the training set for LoRA fine-tuning. The original ``test`` split was reserved and used as the evaluation set to measure performance after fine-tuning.
3.  **Formatting:** Data was formatted to include ``input`` (the headline), ``output`` (the classification label, e.g., "Yes"/"No"), and an appropriate ``instruction`` guiding the model on the headline analysis task.

Named Entity Recognition
~~~~~~~~~~~~~~~~

Similar to Headline Analysis, the data preparation for the Named Entity Recognition (NER) LoRA model utilized the standard splits of the chosen dataset:

1.  **Dataset Used:** The financial Named Entity Recognition (NER) dataset [ner]_ was used.
2.  **Train/Test Split:** The official ``train`` split accompanying the dataset formed the training data for fine-tuning. The corresponding official ``test`` split was used for model evaluation.
3.  **Formatting:** Data was formatted into the required structure, typically involving an ``instruction`` asking for the entity type of a specific phrase within the ``input`` sentence, and the ``output`` being the correct entity label (e.g., "location", "person", "organization").




.. rubric:: Citations

.. [headline] Sinha, A., & Khandait, P. (2020). *Headline-Enhanced Financial Embedding*. In Proceedings of the 2nd Workshop on Economics and Natural Language Processing (pp. 66-74).
.. [ner] Salinas Alvarado, D., Rönnqvist, S., & Niklaus, J. (2015). *Domain-Specific Named Entity Recognition: A Case Study in Finance*. In Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing (pp. 110-115).