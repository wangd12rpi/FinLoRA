Setup
==========

This guide will help you set up the environment for FinLoRA.

GPU Requirements
----------------

FinLoRA works with CUDA-enabled GPUs. CUDA should be at least version 11.8.

GPU memory requirements depend on the size of the LLM, quantization, batch size, and prompt length. For Llama 3.1 8B Instruct, we recommend the following:

- **NVIDIA GPU with at least 24GB VRAM** for 8-bit quantization
- **NVIDIA GPU with at least 16GB VRAM** for 4-bit quantization

RunPod Setup (Optional)
-----------------------

If you don't have access to GPUs with sufficient VRAM, you can rent them affordably from cloud providers like `RunPod <https://www.runpod.io>`_. To create a proper RunPod environment, you can follow these steps:

1. After you have created a RunPod account, go to the "Billing" tab and add $10 of credits. In our testing, when we rented 4 A5000 GPUs, we spent an average of $1.05/hr.

2. Now go click on the "Storage" tab. This tab allows you to create network volumes for persistent storage of uploaded files and models if you disconnect from the service.

3. Click on "New Network Volume" and select a Datacenter that shows that RTX A5000s are available.

4. Name your network volume and make the size of the volume 50 GB. This should only cost $3.50 a month. Then click "Create Network Volume."

5. Under the storage tab, click "Deploy" on your network volume. Select the RTX A5000 GPU.

6. Name your pod, set "GPU Count" to 4, and select the "Runpod Pytorch 2.8.0" pod template. Note: If you only want to run inference instead of fine-tuning, you can select 1.

7. Make sure the instance pricing is set to on-demand. This should cost $0.26/hr per A5000 GPU.

8. Click "Deploy On-Demand."

Package Installation
--------------------

You can set up the environment using either the provided setup script or conda environment file.

Using setup.sh
^^^^^^^^^^^^^^^

The easiest way to set up the environment is to use the provided setup script:

.. code-block:: bash

   git clone https://github.com/Open-Finance-Lab/FinLoRA.git
   cd FinLoRA
   chmod +x setup.sh
   ./setup.sh

This script will install all the required dependencies, including:

- PyTorch with CUDA support
- Transformers library
- Axolotl for fine-tuning
- Other required libraries

Using conda
^^^^^^^^^^^^

Alternatively, you can use the provided conda environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate finenv

Login to Hugging Face
---------------------

When using Llama models, you need to login to Hugging Face due to the LLMs being gated. Run the following command:

.. code-block:: bash

   huggingface-cli login

You will be prompted to enter your Hugging Face token. You can find your token at https://huggingface.co/settings/tokens.

Alternatively, you can set the HF_TOKEN environment variable:

.. code-block:: bash

   export HF_TOKEN=your_token_here
