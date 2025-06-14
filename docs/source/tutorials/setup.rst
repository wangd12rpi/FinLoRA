Setup
==========

This guide will help you set up the environment for FinLoRA.

Environment Setup
----------------

You can set up the environment using either the provided setup script or conda environment file.

Using setup.sh
^^^^^^^^^^^^^

The easiest way to set up the environment is to use the provided setup script:

.. code-block:: bash

   ./setup.sh

This script will install all the required dependencies, including:

- PyTorch with CUDA support
- Transformers library
- Axolotl for fine-tuning
- Other required libraries

Using conda
^^^^^^^^^^

Alternatively, you can use the provided conda environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate finenv

GPU Requirements
---------------

FinLoRA is designed to work with CUDA-enabled GPUs. The code has been tested with CUDA 11.8 and 12.1.

GPU memory depend on the size of the LLM, quantization, batch size, and prompt length. for Llama 3.1 8b generally we recommend:

- NVIDIA GPU with at least 24GB VRAM for 8-bit quantization
- NVIDIA GPU with at least 16GB VRAM for 4-bit quantization

Hugging Face Authentication
--------------------------

When using Llama models, you need to authenticate with Hugging Face:

.. code-block:: bash

   huggingface-cli login

You will be prompted to enter your Hugging Face token. You can find your token at https://huggingface.co/settings/tokens.

Alternatively, you can set the HF_TOKEN environment variable:

.. code-block:: bash

   export HF_TOKEN=your_token_here
