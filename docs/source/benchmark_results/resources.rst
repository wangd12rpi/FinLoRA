Resource Usage and Cost Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tuning Costs
----------------------------

*Part of Benchmark Angle III: Resources of LoRA Fine-tuning and Inference*

The computational requirements for fine-tuning large language models vary dramatically depending on the approach used. The following table compares the cost and time requirements across different methods:

.. list-table:: Fine-tuning Cost Comparison
   :header-rows: 1
   :widths: 25 15 20 20

   * - Models
     - Time
     - GPUs
     - Est. Cost (USD)
   * - BloombergGPT
     - 53 days
     - 512×A100
     - $2.7M
   * - LoRA
     - 14.9h
     - 4×A5000
     - $15.50
   * - QLoRA
     - 14.1h
     - 4×A5000
     - $14.66
   * - DoRA
     - 15.9h
     - 4×A5000
     - $16.54
   * - rsLoRA
     - 14.5h
     - 4×A5000
     - $15.11
   * - Gemini 2.0 FL
     - 8.8h
     - -
     - $162.02
   * - GPT-4o-mini
     - -
     - -
     - $312.00

*Note: GPT-4o cost is estimated based on 4 epochs of fine-tuning at OpenAI fine-tuning pricing.*



LoRA-based methods demonstrate remarkable cost efficiency compared to traditional approaches:

**Self-hosted LoRA Fine-tuning:**
Using four NVIDIA A5000 GPUs, fine-tuning requires 14.1-15.9 hours wall-clock time, corresponding to 56.4-63.6 total GPU hours. At an estimated rate of $0.26 per GPU hour, this translates to approximately $14.66-$16.54 per fine-tuning run.

**Comparison with Commercial Services:**
- LoRA methods: ~$15 (99.5% cost reduction vs. BloombergGPT)
- Gemini 2.0 FL: $162.02 (10× more expensive than LoRA)
- GPT-4o-mini: $312.00 (20× more expensive than LoRA)

This demonstrates that self-hosted LoRA fine-tuning is substantially more cost-effective than commercial fine-tuning services.

GPU Memory Requirements
-----------------------

Memory usage varies significantly based on quantization level and LoRA rank. The following table shows memory requirements for Llama-3.1-8B fine-tuning on NER tasks:

.. list-table:: GPU Memory Usage for Fine-tuning
   :header-rows: 1
   :widths: 30 15 15 15 15 15

   * - Models
     - Parameters
     - GPU Memory (Batch=4)
     - GPU Memory (Batch=8)
     - Model Size
     - Percentage
   * - Llama-3.1-8B-16bit (base)
     - 8.03B
     - -
     - -
     - 16.06 GB
     - -
   * - Llama-3.1-8B-r8-16bit
     - 4.72M
     - 30.91 GB
     - 30.91 GB
     - 16.08 GB
     - 100.1%
   * - Llama-3.1-8B-r8-8bit
     - 4.72M
     - 11.41 GB
     - 11.81 GB
     - 8.04 GB
     - 50.1%
   * - Llama-3.1-8B-r8-4bit
     - 4.72M
     - 8.26 GB
     - 8.65 GB
     - 4.02 GB
     - 25.0%
   * - Llama-3.1-8B-r4-16bit
     - 2.36M
     - 30.90 GB
     - 30.90 GB
     - 16.07 GB
     - 100.1%
   * - Llama-3.1-8B-r4-8bit
     - 2.36M
     - 11.40 GB
     - 11.78 GB
     - 8.03 GB
     - 50.0%
   * - Llama-3.1-8B-r4-4bit
     - 2.36M
     - 8.25 GB
     - 8.61 GB
     - 4.02 GB
     - 25.0%

Memory Usage Insights
---------------

**Quantization Impact:**
- **4-bit quantization**: Reduces memory usage to 25% of the base model
- **8-bit quantization**: Reduces memory usage to ~50% of the base model
- **16-bit (no quantization)**: Maintains full memory requirements

**LoRA Rank Impact:**
The LoRA rank (r4 vs r8) has minimal impact on total memory usage, as the adapter parameters (2.36M vs 4.72M) are negligible compared to the base model size (8.03B parameters).

**Practical Implications:**
- **16-bit fine-tuning**: Requires high-end GPUs (>30GB VRAM)
- **8-bit fine-tuning**: Accessible on mid-range GPUs (12-16GB VRAM)
- **4-bit fine-tuning**: Enables fine-tuning on consumer GPUs (8-12GB VRAM)

**Batch Size Effects:**
Memory usage shows minimal increase when doubling batch size from 4 to 8, indicating efficient gradient accumulation strategies can be employed for larger effective batch sizes.

