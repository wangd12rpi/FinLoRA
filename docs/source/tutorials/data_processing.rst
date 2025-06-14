Data Processing
==============

This guide explains the datasets and data processing scripts used in FinLoRA.

Datasets
--------

FinLoRA uses two main dataset directories:

- ``data/train``: Contains training datasets for fine-tuning models
- ``data/test``: Contains test datasets for evaluating models

Each dataset is stored in JSONL format, with each line containing a JSON object with fields like ``context`` (or ``input``), ``target`` (or ``output``), and sometimes ``instruction``.

Data Processing Scripts
----------------------

FinLoRA includes several scripts for processing raw data into the format required for training and testing:

process_xbrl_extract.py
^^^^^^^^^^^^^^^^^^^^^^^

This script processes XBRL (eXtensible Business Reporting Language) data for extraction tasks. It converts raw XBRL data into a format suitable for training models to extract information from financial reports.

Basic usage:

.. code-block:: bash

   python data/process_xbrl_extract.py --input_file <input_file> --output_file <output_file>

process_finer.py
^^^^^^^^^^^^^^^

This script processes data for the FiNER (Financial Named Entity Recognition) dataset. It prepares data for training models to tag financial entities in text.

Basic usage:

.. code-block:: bash

   python data/process_finer.py --input_file <input_file> --output_file <output_file>

process_fnxl.py
^^^^^^^^^^^^^

This script processes data for the FNXL (Financial XBRL) dataset. It prepares data for training models to work with XBRL tags in financial documents.

Basic usage:

.. code-block:: bash

   python data/process_fnxl.py --input_file <input_file> --output_file <output_file>

Dataset Format
-------------

The processed datasets follow a consistent format:

.. code-block:: json

   {
     "context": "The input text or instruction",
     "target": "The expected output"
   }

or

.. code-block:: json

   {
     "input": "The input text",
     "output": "The expected output",
     "instruction": "The instruction for the model"
   }

These formats are compatible with the training and evaluation scripts used in FinLoRA.