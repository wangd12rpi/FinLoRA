Evaluation
===========

This guide explains how to evaluate fine-tuned models in FinLoRA.

Evaluation Process
----------------

FinLoRA provides scripts to evaluate models on various financial tasks. The evaluation process uses the test datasets in the ``data/test`` directory.

Using run_test.sh
^^^^^^^^^^^^^^^

The main script for evaluation is ``test/run_test.sh``. This script runs ``test.py`` with specific parameters to evaluate models on different datasets.

Basic usage:

.. code-block:: bash

   cd test
   ./run_test.sh

You can modify the script to change the evaluation parameters, such as the dataset, model, and quantization bits.

Using test.py Directly
^^^^^^^^^^^^^^^^^^^

You can also run ``test.py`` directly with custom parameters:

.. code-block:: bash

   python test/test.py \
     --dataset <dataset_name> \
     --base_model <model_path_or_name> \
     --peft_model <peft_model_path> \
     --batch_size <batch_size> \
     --quant_bits <quant_bits> \
     --source <source>

Where:
- ``--dataset``: The dataset to evaluate on (e.g., "sentiment", "headline", "ner")
- ``--base_model``: The base model path or name
- ``--peft_model``: The path to the LoRA adapter (optional)
- ``--batch_size``: Batch size for evaluation
- ``--quant_bits``: Quantization bits (4 or 8)
- ``--source``: The source of the model (e.g., "hf" for Hugging Face, "google" for Google models)

Example:

.. code-block:: bash

   python test/test.py \
     --dataset sentiment \
     --base_model meta-llama/Llama-3.1-8B-Instruct \
     --peft_model lora_adapters/8bits_r8/sentiment_llama_3_1_8b_8bits_r8 \
     --batch_size 8 \
     --quant_bits 8 \
     --source hf

Using run_all_adapters.sh
^^^^^^^^^^^^^^^^^^^^^^^^^

To test multiple adapters systematically, use the ``run_all_adapters.sh`` script:

.. code-block:: bash

   cd test
   bash run_all_adapters.sh

Before running, define the adapters and tasks you want to run in the script by editing the configuration variables. Then execute:

.. code-block:: bash

   bash run_all_adapters.sh

This script allows you to batch evaluate multiple LoRA adapters across different tasks efficiently.

Using run_openai.sh
^^^^^^^^^^^^^^^^^^

To run evaluations using base models from external APIs (e.g., OpenAI):

.. code-block:: bash

   bash run_openai.sh

Before running:

1. Enter your API key in the file
2. Set the tasks you want to run
3. Configure any other API-specific parameters

Then execute:

.. code-block:: bash

   bash run_openai.sh

This is useful for comparing your fine-tuned LoRA adapters against commercial models like GPT-4.

Evaluation Results
---------------

The evaluation results will be printed to the console, including metrics such as accuracy and F1 score. The results can also be found in the ``test/results`` directory.

For each dataset, the evaluation script will generate a report with the model's performance on the test set.

Available Datasets and LoRA Adapters for Evaluation
--------------------------------------------------

The following table lists the available datasets and LoRA adapters for evaluation in FinLoRA:

.. list-table:: Datasets for Evaluation
   :widths: auto
   :header-rows: 1

   * - Dataset
     - Description
     - Dataset Parameter
     - Documentation
   * - Sentiment Analysis
     - Financial sentiment analysis datasets (FPB, FiQA SA, TFNS, NWGI)
     - ``sentiment``
     - :doc:`../tasks/general_financial_tasks`
   * - Headline Analysis
     - Financial headline classification
     - ``headline``
     - :doc:`../tasks/general_financial_tasks`
   * - Named Entity Recognition
     - Financial named entity recognition
     - ``ner``
     - :doc:`../tasks/general_financial_tasks`
   * - FiNER-139
     - XBRL tagging with 139 common US GAAP tags
     - ``finer``
     - :doc:`../tasks/xbrl_reporting_tasks`
   * - XBRL Term
     - XBRL terminology explanation
     - ``xbrl_term``
     - :doc:`../tasks/xbrl_reporting_tasks`
   * - XBRL Extraction
     - Tag and value extraction from XBRL documents
     - ``xbrl_extract``
     - :doc:`../tasks/xbrl_analysis_tasks`
   * - Financial Math
     - Financial mathematics problems
     - ``formula``
     - :doc:`../tasks/xbrl_analysis_tasks`
   * - FinanceBench
     - Financial benchmarking and analysis
     - ``financebench``
     - :doc:`../tasks/xbrl_analysis_tasks`
   * - CFA Level I
     - CFA Level I exam questions
     - ``cfa_level1``
     - :doc:`../tasks/certification_tasks`
   * - CFA Level II
     - CFA Level II exam questions
     - ``cfa_level2``
     - :doc:`../tasks/certification_tasks`
   * - CFA Level III
     - CFA Level III exam questions
     - ``cfa_level3``
     - :doc:`../tasks/certification_tasks`
   * - CPA REG
     - CPA Regulation exam questions
     - ``cpa_reg``
     - :doc:`../tasks/certification_tasks`

.. list-table:: LoRA Adapters for Evaluation
   :widths: auto
   :header-rows: 1

   * - Adapter Type
     - Description
     - Path
     - Documentation
   * - Vanilla LoRA (8-bit)
     - 8-bit quantization with rank 8
     - ``lora_adapters/8bits_r8/<task>_llama_3_1_8b_8bits_r8``
     - :doc:`../lora_methods/lora_methods`
   * - QLoRA (4-bit)
     - 4-bit quantization with rank 4
     - ``lora_adapters/4bits_r4/<task>_llama_3_1_8b_4bits_r4``
     - :doc:`../lora_methods/qlora`
   * - DoRA
     - Weight-Decomposed Low-Rank Adaptation
     - ``lora_adapters/8bits_r8_dora/<task>_llama_3_1_8b_8bits_r8_dora``
     - :doc:`../lora_methods/dora`
   * - RSLoRA
     - Rank-Stabilized LoRA
     - ``lora_adapters/8bits_r8_rslora/<task>_llama_3_1_8b_8bits_r8_rslora``
     - :doc:`../lora_methods/rslora`

Replace ``<task>`` with the specific task name (e.g., ``sentiment``, ``headline``, ``ner``, etc.).
