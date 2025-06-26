Federated Learning with LoRA (FedLoRA)
======================================

This guide explains how to perform federated learning with LoRA (FedLoRA) fine-tuning using the Flower framework in FinLoRA.

Setup and Installation
----------------------

FedLoRA is implemented using the Flower framework, which provides a flexible platform for federated learning.

Navigate to FedLoRA Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, navigate to the FedLoRA directory:

.. code-block:: bash

   cd lora/flowertune-llm

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Install the required dependencies for FedLoRA:

.. code-block:: bash

   pip install -e .

This will install the flowertune-llm package and its dependencies, including the Flower framework.

Running FedLoRA
----------------

Basic FedLoRA Simulation
^^^^^^^^^^^^^^^^^^^^^^^^

Run the FedLoRA simulation with default settings:

.. code-block:: bash

   flwr run .

This command will start a FedLoRA simulation using the default configuration.

Custom Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the FedLoRA configuration using various run-config parameters:

**Using Different Base Models:**

.. code-block:: bash

   # Use OpenLLaMA-7B instead of 3B with 8-bit quantization
   flwr run . --run-config "model.name='openlm-research/open_llama_7b_v2' model.quantization=8"

**Adjusting Training Parameters:**

.. code-block:: bash

   # Run for 50 rounds with 25% client participation
   flwr run . --run-config "num-server-rounds=50 strategy.fraction-fit=0.25"

**Combining Multiple Configuration Options:**

.. code-block:: bash

   # Custom model with extended training
   flwr run . --run-config "model.name='meta-llama/Llama-3.1-8B-Instruct' model.quantization=8 num-server-rounds=30 strategy.fraction-fit=0.3"

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

The following table describes the key configuration parameters for FedLoRA:

.. list-table:: FedLoRA Configuration Parameters
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Description
     - Default Value
     - Example Values
   * - ``model.name``
     - Base model to use for federated training
     - ``openlm-research/open_llama_3b_v2``
     - ``meta-llama/Llama-3.1-8B-Instruct``, ``openlm-research/open_llama_7b_v2``
   * - ``model.quantization``
     - Quantization bits for the model
     - ``4``
     - ``4``, ``8``, ``16``
   * - ``num-server-rounds``
     - Number of FedLoRA rounds
     - ``10``
     - ``10``, ``20``, ``50``
   * - ``strategy.fraction-fit``
     - Fraction of clients participating in each round
     - ``0.1``
     - ``0.1``, ``0.25``, ``0.5``
   * - ``strategy.fraction-evaluate``
     - Fraction of clients used for evaluation
     - ``0.1``
     - ``0.1``, ``0.2``

Related Documentation
---------------------

For more information on LoRA methods and financial tasks, see:

- :doc:`../lora_methods/fed_lora` - Detailed FedLoRA documentation
- :doc:`finetune` - Standard fine-tuning procedures
- :doc:`eval` - Evaluation methods and metrics
- :doc:`../tasks/general_financial_tasks` - Financial task descriptions 
