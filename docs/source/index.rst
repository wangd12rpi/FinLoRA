.. FinLoRA documentation master file, created by
   sphinx-quickstart on Tue Jan 21 19:38:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FinLoRA
==================================================


.. raw:: html

   <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">


.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Introduction

   intro/overview

.. toctree::
   :maxdepth: 2
   :caption: Tasks

   tasks/general_financial_tasks
   tasks/xbrl_reporting_tasks
   tasks/xbrl_analysis_tasks
   tasks/dataset_processing

.. toctree::
   :maxdepth: 3
   :caption: LoRA Methods

   lora_methods/lora_methods
   lora_methods/qlora
   lora_methods/dora
   lora_methods/rslora
   lora_methods/fed_lora


.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   tutorials/setup
   tutorials/finetune
   tutorials/eval

.. toctree::
   :maxdepth: 3
   :caption: Benchmark Results

   benchmark_results/angles
   benchmark_results/results
   benchmark_results/comparisons