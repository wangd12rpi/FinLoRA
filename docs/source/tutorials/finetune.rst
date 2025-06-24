Fine-Tuning
==========

This guide explains how to fine-tune models using LoRA (Low-Rank Adaptation) in FinLoRA.

Fine-Tuning Process
------------------

FinLoRA uses the Axolotl library for fine-tuning, which is wrapped in a convenient script. The fine-tuning process is controlled by configuration files.

Using finetune.py
^^^^^^^^^^^^^^^

The main script for fine-tuning is ``lora/finetune.py``. This script takes a configuration file as input and generates an Axolotl YAML configuration file, which is then used to run the fine-tuning process.

Basic usage:

.. code-block:: bash

   python lora/finetune.py --config lora/finetune_configs.json --run_name <run_name>

Where:
- ``--config`` specifies the path to the configuration file
- ``--run_name`` specifies the name of the configuration to use from the config file

Configuration File
^^^^^^^^^^^^^^^^

The configuration file (``finetune_configs.json``) contains settings for different fine-tuning runs. Each configuration includes:

- ``base_model``: The base model to fine-tune (e.g., "meta-llama/Llama-3.1-8B-Instruct")
- ``dataset_path``: Path to the training dataset
- ``lora_r``: Rank of the LoRA adapters
- ``quant_bits``: Quantization bits (4 or 8)
- ``learning_rate``: Learning rate for training
- ``num_epochs``: Number of training epochs
- ``batch_size``: Batch size for training
- ``gradient_accumulation_steps``: Number of gradient accumulation steps
- Additional parameters for specific LoRA variants (e.g., ``peft_use_rslora``, ``peft_use_dora``)

Example configuration:

.. code-block:: json

   {
     "sentiment_llama_3_1_8b_8bits_r8": {
       "base_model": "meta-llama/Llama-3.1-8B-Instruct",
       "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
       "lora_r": 8,
       "quant_bits": 8,
       "learning_rate": 0.0001,
       "num_epochs": 4,
       "batch_size": 8,
       "gradient_accumulation_steps": 2
     }
   }

LoRA Adapters
-----------

The fine-tuned LoRA adapters are saved in the ``lora_adapters`` directory. This directory contains subdirectories for different quantization and rank configurations:

- ``lora_adapters/4bits_r4``: 4-bit quantization with rank 4
- ``lora_adapters/8bits_r8``: 8-bit quantization with rank 8
- ``lora_adapters/8bits_r8_dora``: 8-bit quantization with rank 8 using DoRA
- ``lora_adapters/8bits_r8_rslora``: 8-bit quantization with rank 8 using RSLoRA

Each subdirectory contains the fine-tuned adapters for different tasks, such as sentiment analysis, headline analysis, named entity recognition, etc.

These adapters can be loaded during evaluation to test the fine-tuned models.

Available Datasets and LoRA Methods
----------------------------------

The following table lists the available datasets and LoRA methods in FinLoRA, along with links to their documentation:

.. list-table:: Datasets and LoRA Methods
   :widths: auto
   :header-rows: 1

   * - Dataset
     - Description
     - Configuration Name
     - Documentation
   * - Sentiment Analysis
     - Financial sentiment analysis datasets (FPB, FiQA SA, TFNS, NWGI)
     - ``sentiment_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/general_financial_tasks`
   * - Headline Analysis
     - Financial headline classification
     - ``headline_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/general_financial_tasks`
   * - Named Entity Recognition
     - Financial named entity recognition
     - ``ner_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/general_financial_tasks`
   * - FiNER-139
     - XBRL tagging with 139 common US GAAP tags
     - ``finer_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/xbrl_reporting_tasks`
   * - XBRL Term
     - XBRL terminology explanation
     - ``xbrl_term_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/xbrl_reporting_tasks`
   * - XBRL Extraction
     - Tag and value extraction from XBRL documents
     - ``xbrl_extract_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/xbrl_analysis_tasks`
   * - Financial Math
     - Financial mathematics problems
     - ``formula_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/xbrl_analysis_tasks`
   * - FinanceBench
     - Financial benchmarking and analysis
     - ``financebench_llama_3_1_8b_8bits_r8``
     - :doc:`../tasks/xbrl_analysis_tasks`

.. list-table:: LoRA Methods
   :widths: auto
   :header-rows: 1

   * - Method
     - Description
     - Configuration Parameter
     - Documentation
   * - Standard LoRA
     - Low-Rank Adaptation
     - No special parameter needed
     - :doc:`../lora_methods/lora_methods`
   * - QLoRA
     - Quantized LoRA (4-bit)
     - ``quant_bits: 4``
     - :doc:`../lora_methods/qlora`
   * - DoRA
     - Weight-Decomposed Low-Rank Adaptation
     - ``peft_use_dora: true``
     - :doc:`../lora_methods/dora`
   * - RSLoRA
     - Rank-Stabilized LoRA
     - ``peft_use_rslora: true``
     - :doc:`../lora_methods/rslora`
   * - FedLoRA
     - Federated Learning with LoRA
     - Uses Flower framework
     - :doc:`../lora_methods/fed_lora`
