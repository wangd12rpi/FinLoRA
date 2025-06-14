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

LoRA Adaptors
-----------

The fine-tuned LoRA adapters are saved in the ``lora_adaptors`` directory. This directory contains subdirectories for different quantization and rank configurations:

- ``lora_adaptors/4bits_r4``: 4-bit quantization with rank 4
- ``lora_adaptors/8bits_r8``: 8-bit quantization with rank 8
- ``lora_adaptors/8bits_r8_dora``: 8-bit quantization with rank 8 using DoRA
- ``lora_adaptors/8bits_r8_rslora``: 8-bit quantization with rank 8 using RSLoRA

Each subdirectory contains the fine-tuned adapters for different tasks, such as sentiment analysis, headline analysis, named entity recognition, etc.

These adapters can be loaded during evaluation to test the fine-tuned models.
