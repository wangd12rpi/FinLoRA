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

Evaluation Results
---------------

The evaluation results will be printed to the console, including metrics such as accuracy and F1 score. The results can also be found in the ``test/results`` directory.

For each dataset, the evaluation script will generate a report with the model's performance on the test set.
