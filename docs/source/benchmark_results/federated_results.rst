Results on Federated LoRA
===============

*Part of Benchmark Angle IV: Data Privacy in Collaborative Training*


Performance Comparison: Central vs FedLoRA
-------------------------------------------


The sensitive nature of financial data necessitates privacy-preserving techniques like Federated Learning for collaborative training. We evaluated federated learning with LoRA (FedLoRA) in a four-node environment using the FedAvg algorithm, where sentiment analysis datasets were partitioned across nodes.

.. list-table:: Performance Comparison on Sentiment Analysis Tasks
   :header-rows: 2
   :widths: 20 20 20 20 20

   * - **Llama 3.1 8B 8bit-r8**
     - FPB
     - FiQA SA
     - TFNS
     - NWGI
   * - 
     - 
     - 
     - 
     - 
   * - Base Model
     - Acc: 68.73% | F1: 0.677
     - Acc: 46.55% | F1: 0.557
     - Acc: 69.97% | F1: 0.683
     - Acc: 46.58% | F1: 0.412
   * - Central LoRA
     - **Acc: 89.11% | F1: 0.941**
     - **Acc: 88.09% | F1: 0.923**
     - **Acc: 91.96% | F1: 0.955**
     - **Acc: 61.92% | F1: 0.748**

Key Findings
--------------

**Performance Hierarchy:**
- Central LoRA achieves the highest performance across all tasks
- FedLoRA shows substantial improvement over base model
- Privacy preservation comes with performance trade-offs

**Task-Specific Results:**
- **TFNS**: Best overall performance (91.96% accuracy, 0.955 F1)
- **FPB & FiQA SA**: Strong performance (~89% accuracy, ~0.93 F1)
- **NWGI**: Most challenging task (61.92% accuracy, 0.748 F1)

**Privacy vs Performance Trade-off:**
While FedLoRA does not match centralized performance levels, it demonstrates notable improvements over the base model while maintaining data privacy across