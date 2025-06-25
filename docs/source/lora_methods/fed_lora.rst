
FedLoRA
~~~~~~~

In the finance sector, multiple banks may want to work together on a model to predict credit risk and whether a borrower will default on a loan. Each bank may have a different dataset, but they cannot share their data due to compliance reasons and privacy concerns. Federated learning solves this issue by fine-tuning a model on local data and aggregating updates during backpropagation to a centralized model via a server.

Differentially Private Low-Rank Adaptation (DP-LoRA) [5]_ is a method to use federated learning with LoRA (FedLoRA).

DP-LoRA first uses a server to send the current global LoRA weights (the A and B matrices from earlier) to all clients.

Every client does the following: 1) Gets a minibatch of its private data 2) Computes the gradient for only its local A and B weights clipped with an :math:`\ell_2` norm (square root of the sum of the squares of elements in the vector) 3) Adds Gaussian noise to the gradients 4) Updates the A and B LoRA matrices 5) Sends the updated A and B matrices to the server.

By adding noise, DP-LoRA prevents the centralized model from inferring the private data later on. This would allow the banks in the credit risk example to work on a model together.

As in normal federated learning, the server then aggregates the weights from all clients in a weighted average and sends the updated weights to all clients.

Using FedLoRA in FinLoRA
------------------------

FinLoRA implements FedLoRA using the Flower framework. To use FedLoRA, you need to navigate to the flowertune-llm directory and run the FedLoRA simulation:

.. code-block:: bash

   cd lora/flowertune-llm
   pip install -e .
   flwr run .

You can customize the configuration using command-line arguments:

.. code-block:: bash

   # Use Llama-3.1-8B instead of the default model and 8-bits quantization
   flwr run . --run-config "model.name='meta-llama/Llama-3.1-8B-Instruct' model.quantization=8"

   # Run for 50 rounds with 25% client participation
   flwr run . --run-config "num-server-rounds=50 strategy.fraction-fit=0.25"

The implementation is located in the ``lora/flowertune-llm`` directory. The main components are:

- ``flowertune_llm/dataset.py``: Handles dataset loading and partitioning for FedLoRA
- ``flowertune_llm/client.py``: Implements the FedLoRA client
- ``flowertune_llm/server.py``: Implements the FedLoRA server

For more details on the implementation, refer to the code in the ``lora/flowertune-llm`` directory.
