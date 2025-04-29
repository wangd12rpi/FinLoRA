==================
XBRL Data Reporting
==================

Overview
************



.. list-table:: Question/training sets for general financial tasks.
   :widths: auto
   :header-rows: 1

   * - Question sets
     - Type
     - #Train
     - #Test
     - Metrics
     - Source
   * - FiNER-139
     - XBRL Tagging
     -
     -
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/TheFinAI/en-fpb>`__
   * - FNXL
     - XBRL Tagging
     -
     -
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification>`__

Tasks Details
************************


This section provides examples for various general financial datasets commonly used in NLP tasks.

FiNER-139
--------------------
Financial Phrase Bank (FPB) contains sentences extracted from financial news and reports. These sentences are annotated with sentiment labels "positive", "negative", and "neutral". We manually created the train/test split.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Input**
     - What is the appropriate XBRL US GAAP tag for "12,028" in the given sentence? Output the US GAAP tag only and nothing else. "Rent expense for this lease amounted to 12,028 and 12,500 for the six months ended January 31 , 2020 and 2019 , respectively ."
   * - **Output**
     - LeaseAndRentalExpense


FNXL
--------------------

Citations
****************
