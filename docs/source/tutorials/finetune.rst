Fine-Tuning
==========

This guide explains how to fine-tune models using LoRA (Low-Rank Adaptation) in FinLoRA.

Fine-Tuning Process
------------------

FinLoRA uses the Axolotl library for fine-tuning, which is wrapped in a convenient script. The fine-tuning process involves several steps outlined below.

Step-by-Step Fine-Tuning
^^^^^^^^^^^^^^^^^^^^^^^

1. **Navigate to the LoRA directory and fetch deepspeed configs**

   First, navigate to the lora directory and fetch deepspeed configs. The deepspeed configs allow the fine-tuning framework to parallelize fine-tuning across GPUs:

   .. code-block:: bash

      cd lora
      axolotl fetch deepspeed_configs

2. **Add your fine-tuning dataset**

   Add your fine-tuning dataset (e.g., ``your_dataset_train.jsonl``) in the ``../data/train/`` folder.

3. **Configure your LoRA adapter**

   Open ``finetune_configs.json`` and add the configuration for the LoRA adapter you want to create with hyperparameters defined. There are examples you can reference in the file. The following is an example:

   .. code-block:: json

      "your_config_name": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_path": "../data/train/your_dataset_train.jsonl",
        "lora_r": 8,
        "quant_bits": 8,
        "learning_rate": 0.0001,
        "num_epochs": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 2
      }

4. **Run fine-tuning**

   Run fine-tuning with your configuration by executing the following command:

   .. code-block:: bash

      python finetune.py your_config_name

   For example, to use the existing formula configuration:

.. code-block:: bash

      python finetune.py formula_llama_3_1_8b_8bits_r8

5. **Retrieve your adapter**

   After fine-tuning completes, the adapter will be saved in the ``axolotl-output`` subfolder within the 'lora' folder. Download the adapter files from this directory. You can remove checkpoints to save space.

.. note::
   If you don't have compute resources, you can rent 4 A5000s at a low cost from `RunPod <https://www.runpod.io>`_.

Configuration File Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Example configurations for different LoRA methods:

**Vanilla LoRA:**

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

**QLoRA (Quantized LoRA):**

.. code-block:: json

   {
     "xbrl_term_llama_3_1_8b_4bits_r4": {
       "base_model": "meta-llama/Llama-3.1-8B-Instruct",
       "dataset_path": "../data/train/xbrl_term_train.jsonl",
       "lora_r": 4,
       "quant_bits": 4,
       "learning_rate": 0.0001,
       "num_epochs": 1,
       "batch_size": 4,
       "gradient_accumulation_steps": 2
     }
   }

**DoRA (Weight-Decomposed Low-Rank Adaptation):**

.. code-block:: json

   {
     "sentiment_llama_3_1_8b_8bits_r8_dora": {
       "base_model": "meta-llama/Llama-3.1-8B-Instruct",
       "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
       "lora_r": 8,
       "quant_bits": 8,
       "learning_rate": 0.0001,
       "num_epochs": 4,
       "batch_size": 8,
       "gradient_accumulation_steps": 2,
       "peft_use_dora": true
     }
   }

**RSLoRA (Rank-Stabilized LoRA):**

.. code-block:: json

   {
     "sentiment_llama_3_1_8b_8bits_r8_rslora": {
       "base_model": "meta-llama/Llama-3.1-8B-Instruct",
       "dataset_path": "../data/train/finlora_sentiment_train.jsonl",
       "lora_r": 8,
       "quant_bits": 8,
       "learning_rate": 0.0001,
       "num_epochs": 4,
       "batch_size": 8,
       "gradient_accumulation_steps": 2,
       "peft_use_rslora": true
     }
   }

Using Your LoRA Adapter
----------------------

Once you have trained a LoRA adapter, you can use it for inference by using the following code:

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel
   import torch

   # Load base model and tokenizer
   base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(base_model_name)
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       torch_dtype=torch.float16,
       device_map="auto",
       trust_remote_code=True
   )

   # Load and apply the LoRA adapter
   adapter_path = "./path/to/your/adapter"  # Path to your adapter
   model = PeftModel.from_pretrained(base_model, adapter_path)

   # Generate text
   prompt = "What is the formula for the Black-Scholes model?"
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

   with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_new_tokens=512,
           # This ensures that you get reproducible responses.
           temperature=0,
           pad_token_id=tokenizer.eos_token_id
       )

   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)

LoRA Adapters Directory Structure
-------------------------------

The fine-tuned LoRA adapters are saved in the ``lora_adapters`` directory. This directory contains subdirectories for different quantization and rank configurations:

- ``lora_adapters/4bits_r4``: 4-bit quantization with rank 4
- ``lora_adapters/8bits_r8``: 8-bit quantization with rank 8
- ``lora_adapters/8bits_r8_dora``: 8-bit quantization with rank 8 using DoRA
- ``lora_adapters/8bits_r8_rslora``: 8-bit quantization with rank 8 using RSLoRA
- ``lora_adapters/fp16_r8``: FP16 precision with rank 8

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
   * - Vanilla LoRA
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
