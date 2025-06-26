FedLoRA
=======

.. contents:: Table of Contents

Background
----------

**Citation:** `Differentially Private Low-Rank Adaptation of Large Language Model Using Federated Learning (Liu et al., 2024) <https://arxiv.org/abs/2312.17493>`_

In the finance sector, multiple institutions may want to collaborate using their own proprietary datasets, but they cannot share their data due to compliance reasons and privacy concerns. Federated learning with LoRA (FedLoRA) solves this issue by fine-tuning a model on local data and aggregating LoRA updates to a central node. This combines the parameter efficiency of LoRA with federated learning principles, enabling organizations to collaborate without exposing sensitive information.

Quick Facts
~~~~~~~~~~~

#. FedLoRA enables multiple financial institutions to collaborate on models without sharing their private data.
#. FedLoRA aggregates only LoRA weights (A and B matrices) rather than full model parameters.
#. DP-LoRA, an implementation of FedLoRA, adds Gaussian noise to gradients to prevent the centralized model from inferring private data.

Algorithmic Idea
~~~~~~~~~~~~~~~~

In the finance sector, multiple banks may want to work together on a model to predict credit risk and whether a borrower will default on a loan. Each bank may have a different dataset, but they cannot share their data due to compliance reasons and privacy concerns. Federated learning solves this issue by fine-tuning a model on local data and aggregating updates during backpropagation to a centralized model via a server.

Differentially Private Low-Rank Adaptation (DP-LoRA) is an implementation of FedLoRA.

DP-LoRA first uses a server to send the current global LoRA weights (the A and B matrices from earlier) to all clients.

Every client does the following: 1) Gets a minibatch of its private data 2) Computes the gradient for only its local A and B weights clipped with an :math:`\ell_2` norm (square root of the sum of the squares of elements in the vector) 3) Adds Gaussian noise to the gradients 4) Updates the A and B LoRA matrices 5) Sends the updated A and B matrices to the server.

By adding noise, DP-LoRA prevents the centralized model from inferring the private data later on. This would allow the banks in the credit risk example to work on a model together.

As in normal federated learning, the server then aggregates the weights from all clients in a weighted average and sends the updated weights to all clients.

Key Equations
~~~~~~~~~~~~~

For a federated learning setup with :math:`K` clients, the FedLoRA update process follows:

.. math::

   \mathbf{A}^{(t+1)}_{\text{global}}, \mathbf{B}^{(t+1)}_{\text{global}} = \text{FedAvg}\left(\{\mathbf{A}^{(t)}_k, \mathbf{B}^{(t)}_k\}_{k=1}^K\right)

Where the federated averaging is:

.. math::

   \mathbf{A}^{(t+1)}_{\text{global}} = \sum_{k=1}^K \frac{n_k}{n} \mathbf{A}^{(t)}_k, \quad \mathbf{B}^{(t+1)}_{\text{global}} = \sum_{k=1}^K \frac{n_k}{n} \mathbf{B}^{(t)}_k

For differential privacy, the DP-LoRA gradient update includes noise addition:

.. math::

   \tilde{\nabla}\mathbf{A}_k = \text{Clip}(\nabla\mathbf{A}_k, C) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})

Where:

#. :math:`n_k` is the number of samples at client :math:`k` and :math:`n = \sum_{k=1}^K n_k`.
#. :math:`\text{Clip}(\cdot, C)` performs :math:`\ell_2` norm clipping with threshold :math:`C`.
#. :math:`\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})` is Gaussian noise for differential privacy.
#. :math:`\sigma` is the noise multiplier controlling the privacy-utility tradeoff.

Implementation in FinLoRA
~~~~~~~~~~~~~~~~~~~~~~~~~

FinLoRA integrates with the Flower federated learning framework to enable FedLoRA training. The implementation supports both standard federated LoRA and differential privacy through DP-LoRA.

Configuration example for federated training:

.. code-block:: yaml

   # FedLoRA configuration
   federated_learning: true
   use_dp_lora: true
   
   # Privacy parameters
   dp_epsilon: 1.0
   dp_delta: 1e-5
   noise_multiplier: 1.0
   max_grad_norm: 1.0
   
   # Standard LoRA parameters
   lora_r: 8
   lora_alpha: 16
   quant_bits: 8
   
   # Federated learning parameters
   num_clients: 4
   client_fraction: 1.0
   num_rounds: 10

Key parameters:
- ``federated_learning``: Enable federated training mode
- ``use_dp_lora``: Enable differential privacy guarantees
- ``dp_epsilon``: Privacy budget (lower = more private)
- ``noise_multiplier``: Controls the amount of noise added to gradients

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from flowertune_llm import client_app, server_app
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch

   # Initialize federated learning components
   def create_federated_model():
       base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
       model = AutoModelForCausalLM.from_pretrained(
           base_model_name,
           torch_dtype=torch.float16,
           device_map="auto"
       )
       return model

   # Client-side training with local data
   def train_client(model, local_data, client_id):
       # Local LoRA fine-tuning with DP-LoRA
       from peft import LoraConfig, get_peft_model
       
       lora_config = LoraConfig(
           r=8,
           lora_alpha=16,
           target_modules=["q_proj", "v_proj"],
           lora_dropout=0.1,
           use_rslora=False,  # Can combine with other LoRA variants
       )
       
       model = get_peft_model(model, lora_config)
       
       # Train with differential privacy
       # (Implementation details handled by Flower framework)
       
       return model.state_dict()

   # Server aggregation
   def aggregate_weights(client_weights):
       # Federated averaging of LoRA adapters
       # (Handled automatically by Flower server)
       pass

   # Run federated training
   # flower-simulation --app-dir=lora/flowertune-llm --num-supernodes=4

References
----------

.. [1] Liu, X. Y., Zhu, R., Zha, D., Gao, J., Zhong, S., White, M., & Qiu, M. (2024). Differentially private low-rank adaptation of large language model using federated learning. *arXiv preprint arXiv:2312.17493*.

Why This Method?
~~~~~~~~~~~~~~~

FedLoRA is essential for the financial sector to collaborate on models without sharing their private data, which may be proprietary or regulated. DP-LoRA is an useful implementation of FedLoRA that adds noise to the gradients to prevent the centralized model from inferring the private data, which would leak it.

Useful Links
~~~~~~~~~~~~

* `DP-LoRA Official Implementation <https://github.com/LLM-Data-Privacy/DP-LoRA>`_ - Reference implementation of Differentially Private LoRA
* `Flower Federated Learning Framework <https://flower.ai>`_ - Open-source federated learning framework used in FinLoRA
* `FlowerTune LLM Tutorial <https://flower.ai/docs/examples/flowertune-llm.html>`_ - Comprehensive guide for federated LLM fine-tuning
* `Flower Documentation <https://flower.ai/docs/>`_ - Complete documentation for federated learning with Flower
