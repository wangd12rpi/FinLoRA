Data Processing
==============

This guide explains the datasets and data processing scripts used in FinLoRA.

Dataset Overview
----------------

FinLoRA tests Llama 3.1 8B Instruct with LoRA adapters on 19 datasets across 4 different types of tasks, ranging from general financial tasks to professional level XBRL (eXtensible Business Reporting Language)-based financial statement analysis.

The train-test splits for the four task categories are as follows:

- **General Financial Tasks**: 122.9k/31.7k
- **Financial Certificate Tasks**: 472/346  
- **Financial Reporting Tasks**: 15.9k/8.3k
- **Financial Statement Analysis Tasks**: 27.9k/7.3k

Dataset Categories
-----------------

**General Financial Tasks:**

- **Sentiment Analysis (FPB, FiQA SA, TFNS)**: Financial sentences classified with sentiment from ``{negative, neutral, positive}``
- **NWGI Sentiment**: Financial text classified into 7-level sentiment, simplified to ``{negative, neutral, positive}``
- **Headline Analysis**: Financial headlines classified with binary answers from ``{Yes, No}``
- **Named Entity Recognition**: Financial text with highlighted entities classified into ``{person, location, organization}``

**Financial Certificate Tasks:**

- **CFA Level I/II/III & CPA REG**: Multiple choice questions from mock exams with answers from ``{A, B, C, D}`` or ``{A, B, C}``

**Financial Reporting Tasks:**

- **XBRL Term**: Brief explanations for XBRL terminology from XBRL International website
- **FiNER/FNXL Tagging**: Financial text with numerical entities tagged with appropriate US GAAP tags

**Financial Statement Analysis Tasks:**

- **XBRL Tag Extraction**: XBRL context analysis to identify specific XBRL tags
- **XBRL Value Extraction**: XBRL context analysis to find specific numerical values
- **XBRL Formula Construction**: Creating financial formulas using US GAAP tags
- **XBRL Formula Calculation**: Substituting numerical values into financial formulas
- **Financial Math**: Applying financial formulas to solve numerical problems
- **FinanceBench**: Answering questions based on XBRL financial reports

Dataset Directories
-------------------

FinLoRA uses two main dataset directories:

- ``data/train``: Contains training datasets for fine-tuning models
- ``data/test``: Contains test datasets for evaluating models

Each dataset is stored in JSONL format, with each line containing a JSON object with fields like ``context`` (or ``input``), ``target`` (or ``output``), and sometimes ``instruction``.

Dataset Formats
---------------

The processed datasets follow consistent formats:

**Standard Format:**

.. code-block:: json

   {
     "context": "The input text/instruction",
     "target": "The expected output"
   }

Data Processing Scripts
----------------------

FinLoRA includes several scripts for processing raw data into the format required for training and testing:

process_xbrl_extract.py
^^^^^^^^^^^^^^^^^^^^^^^

This script processes XBRL (eXtensible Business Reporting Language) data for extraction tasks. It converts raw XBRL data into a format suitable for training models to extract information from financial reports.

.. code-block:: bash

   python data/process_xbrl_extract.py

The script processes multiple categories:
- Tags extraction
- Value extraction  
- Formula construction
- Formula calculations

process_finer.py
^^^^^^^^^^^^^^^

This script processes data for the FiNER-139 (Financial Named Entity Recognition) dataset. It prepares data for training models to tag financial entities with 139 common US GAAP tags.

.. code-block:: bash

   python data/process_finer.py

process_fnxl.py
^^^^^^^^^^^^^

This script processes data for the FNXL (Financial XBRL) dataset. It prepares batched data for training models to work with XBRL tags in financial documents.

.. code-block:: bash

   python data/process_fnxl.py

process_sentiment_train.py
^^^^^^^^^^^^^^^^^^^^^^^^^^

This script processes various financial sentiment analysis datasets including FPB, FiQA SA, TFNS, and NWGI.

.. code-block:: bash

   python data/process_sentiment_train.py

The script handles:
- Financial Phrasebank (FPB) sentiment classification
- FiQA sentiment analysis
- Twitter Financial News Sentiment (TFNS)
- News With GPT Instruction (NWGI) sentiment

process_xbrl_agent_data.py
^^^^^^^^^^^^^^^^^^^^^^^^^^

This script processes XBRL terminology data and FinanceBench datasets with OCR capabilities for PDF processing.

.. code-block:: bash

   python data/process_xbrl_agent_data.py

It does the folowing:
- XBRL terminology processing
- FinanceBench data with PDF OCR
- Formula data processing

process_multitask_data.py
^^^^^^^^^^^^^^^^^^^^^^^^^

This script processes data for multi-task learning scenarios, combining multiple financial datasets.

.. code-block:: bash

   python data/process_multitask_data.py

Additional Processing Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``convert_gemini_format.py``: Converts datasets to Gemini API format
- ``dataset_statistics.py``: Computes dataset statistics and metrics
- ``process_xbrl_formula.py``: Processes XBRL formula-specific data
- ``process_xbrl_tag.py``: Processes XBRL tag extraction data

Dataset Statistics
-----------------

The datasets vary significantly in size and complexity:

**Average Prompt Lengths:**
- Certificate tasks (CFA/CPA): 147-1,000 tokens
- XBRL Analysis tasks: 3,800+ tokens
- General Financial tasks: 43-138 tokens
- Reporting tasks: 25-7,100 tokens

**Evaluation Metrics:**
- Most datasets: Accuracy and F1 score
- XBRL Term and FinanceBench: BERTScore F1
- Complex reasoning tasks: Custom evaluation metrics

Working with Custom Datasets
----------------------------

To add your own financial dataset:

1. Prepare your data in the required JSONL format
2. Place training data in ``data/train/``
3. Place test data in ``data/test/``
4. Update configuration in ``finetune_configs.json``
5. Run processing scripts if needed

Example dataset entry:

.. code-block:: json

   {
     "context": "Instruction: Analyze the financial statement...\nInput: Company XYZ reported...\nAnswer: ",
     "target": "The company shows strong performance..."
   }


Related Documentation
--------------------

For more information on specific tasks and evaluation methods, see:

- :doc:`../tasks/general_financial_tasks` - General financial task descriptions
- :doc:`../tasks/certification_tasks` - Professional certification tasks  
- :doc:`../tasks/xbrl_reporting_tasks` - XBRL reporting and tagging
- :doc:`../tasks/xbrl_analysis_tasks` - XBRL analysis and extraction
- :doc:`finetune` - Fine-tuning with processed datasets
- :doc:`eval` - Evaluation methods and metrics