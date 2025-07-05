Financial Data Reporting
==================

XBRL tagging is a crucial step in creating XBRL reports. This process involves tagging numerical entities within texts, such as earnings call transcripts, using US GAAP tags.

.. list-table:: Question/training sets for general financial tasks.
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
   * - FiNER-139
     - XBRL Tagging
     - 10.0k
     - 7.4k
     - 1.8k
     - Accuracy, F1
     - `Hugging Face <https://huggingface.co/datasets/nlpaueb/finer-139>`__
     - `finer_train_batched.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/finer_train_batched.jsonl>`__
     - `finer_test_batched.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/finer_test_batched.jsonl>`__
   * - FNXL
     - XBRL Tagging
     - -
     - 247
     - 7.1k
     - Accuracy, F1
     - `GitHub <https://github.com/soummyaah/FNXL>`__
     - Test only dataset
     - `fnxl_test_batched.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/fnxl_test_batched.jsonl>`__
   * - XBRL Term
     - Terminology
     - 5.9k
     - 651
     - 25
     - BERTScore
     - `GitHub <https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/XBRL%20Terminology.xlsx>`__
     - `xbrl_term_train.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/train/xbrl_term_train.jsonl>`__
     - `xbrl_term_test.jsonl <https://github.com/Open-Finance-Lab/FinLoRA/blob/main/data/test/xbrl_term_test.jsonl>`__

Two distinct approaches are employed:

* **Regular Approach**: The input addresses only one numerical entity at a time, and a list of potential US GAAP tag options is not provided. This approach is designed for simple and quick queries. Since options aren't provided, input processing can be faster. Fine-tuned models perform well after learning the tag taxonomy, but base models perform poorly because they lack knowledge of all the valid tags.

* **Batched Approach**: The input includes multiple (e.g., four) numerical entities to be tagged simultaneously, and a list of potential US GAAP tag options is provided. Providing the options allows the Large Language Model (LLM) to know which tags are valid choices without needing pre-existing knowledge of the entire taxonomy; it can infer the appropriate tag from its name and context. To improve efficiency and reduce token usage, given the large number of tags, multiple tagging questions are grouped into a single input batch following the list of options.

FiNER-139
--------------------
FiNER includes XBRL Tagging tasks with labels from the 139 most common US GAAP tags.

**Regular Approach:**

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - What is the appropriate XBRL US GAAP tag for the specified numerical entity in the given sentence? Output the US GAAP tag only and nothing else.
   * - **Input**
     - "Rent expense for this lease amounted to 12,028 and 12,500 for the six months ended January 31 , 2020 and 2019 , respectively ." Entity: "12,028"
   * - **Output**
     - LeaseAndRentalExpense

**Batched Approach:**

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are an XBRL expert. Here is a list of US GAAP tags options. Answer the following 4 independent questions by providing only 4 US GAAP tags answers in the order of the questions. Each answer must be separated by a comma (,). Provide nothing else.
   * - **Input**
     - US GAAP tags options: SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage, InterestExpense, ... [omitted for this table]

       1. What is best tag for entity "0.36" in sentence: "On May 27 , 2020 , the Company's Board of Directors declared a quarterly cash dividend of $ 0.36 per share , which is payable on or before July 21 , 2020 to shareholders of record on July 7 , 2020 ."?

       2. What is best tag for entity "114" in sentence: "As of March 31 , 2020 , we owned interests in the following assets : 116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ; 90 hotel condominium units at World Quest Resort in Orlando , Florida ( World Quest ) ; and 17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels ."?

       3. What is best tag for entity "two" in sentence: "As of March 31 , 2020 , we owned interests in the following assets : 116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ; 90 hotel condominium units at World Quest Resort in Orlando , Florida ( World Quest ) ; and 17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels ."?

       4. What is best tag for entity "2.8" in sentence: "As of March 31 , 2020 , we owned interests in the following assets : 116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ; 90 hotel condominium units at World Quest Resort in Orlando , Florida ( World Quest ) ; and 17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels ."?
   * - **Output**
     - CommonStockDividendsPerShareDeclared,NumberOfRealEstateProperties,NumberOfRealEstateProperties,EquityMethodInvestments

FNXL
--------------------
FNXL is similar to FiNER, but includes even more US GAAP tags as labels.

**Regular Approach:**

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are an XBRL expert. Here is a list of US GAAP tags options. What is the best US GAAP tag for the specified entity in the given sentence?
   * - **Input**
     - US GAAP tags options: [omitted for this table]

       Sentence: "The 2018 ASR Agreement was completed on January 29, 2019, at which time the Company received 117,751 additional shares based on a final weighted average per share purchase price during the repurchase period of $187.27."

       Entity: "187.27"
   * - **Output**
     - AcceleratedShareRepurchasesFinalPricePaidPerShare

**Batched Approach:**

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are an XBRL expert. Here is a list of US GAAP tags options. Choose the best XBRL US GAAP tag for each highlighted entity in the sentences below. Provide only the US GAAP tags, comma-separated, in the order of the sentences and highlighted entity. Provide nothing else.
   * - **Input**
     - US GAAP tags options: [omitted for this table]

       What is the best US GAAP tag for entity "6.3" in sentence: "The projected benefit obligation and fair value of plan assets for U.S. pension plans with projected benefit obligations in excess of plan assets was $6.3 billion and $4.7 billion, respectively, as of December31, 2019 and $5.5 billion and $4.1 billion, respectively, as of December31, 2018."?

       What is the best US GAAP tag for entity "124,043" in sentence: "Capitalized software, net of accumulated amortization of $124,043 in 2020 and $104,237 in 2019"?

       What is the best US GAAP tag for entity "1.5" in sentence: "The Company purchased 30 million and 57 million shares under stock repurchase programs in fiscal 2020 and 2019 at a cost of $1.5 billion and $3.8 billion, respectively."?

       What is the best US GAAP tag for entity "651,313" in sentence: "This multi-tenant mortgage loan is interest-only with a principal balance due on maturity, and it is secured by seven properties in six states, totaling approximately 651,313 square feet."?
   * - **Output**
     - DefinedBenefitPlanPensionPlanWithProjectedBenefitObligationInExcessOfPlanAssetsProjectedBenefitObligation,CapitalizedComputerSoftwareAccumulatedAmortization,PaymentsForRepurchaseOfCommonStock,AreaOfRealEstateProperty


Fine-tuning for Financial Reporting Tasks
--------------------------------------------------

To fine-tune a model for financial reporting tasks, you can use the configurations provided in the ``lora/finetune_configs.json`` file. Below are the configurations for each task:

FiNER-139
^^^^^^^^^^^^^

To fine-tune a model for the FiNER-139 task, you can use one of the following configurations:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py finer_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py finer_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py finer_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py finer_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **finer_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **finer_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **finer_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **finer_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.

XBRL Term
^^^^^^^^^^^^^

To fine-tune a model for the XBRL Term task, you can use one of the following configurations:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py xbrl_term_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py xbrl_term_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py xbrl_term_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py xbrl_term_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **xbrl_term_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **xbrl_term_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **xbrl_term_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **xbrl_term_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.
