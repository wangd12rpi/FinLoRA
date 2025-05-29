=======
Benchmark Results
=======

Angle I: LoRA Methods Performance on Financial Datasets
-------------------------------------------------------

Average performance of base models and LoRA models:

.. raw:: html

    <object style="width: 100%" data="../_static/images/p1_new.svg" type="image/svg+xml"></object>

Performance on financial tasks:

.. raw:: html

    <object style="width: 100%" data="../_static/images/full_results.svg" type="image/svg+xml"></object>


Comparative Performance of LoRA Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The performance of base models and different LoRA fine-tuned models is detailed in the main results tables. Vanilla LoRA (8-bit, rank 8) achieves the highest overall average score (74.74), a 37.69% increase over the Llama 3.1 8B base model's 37.05. The performance breakdown by category, often illustrated in accompanying figures, shows that Vanilla LoRA outperforms other LoRA variants in general financial tasks, while rsLoRA leads in financial analysis, financial reporting, and financial statement analysis.

rsLoRA Performs Better at High Ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rsLoRA scales with :math:`\alpha/\sqrt{r}` instead of :math:`\alpha/r` to prevent gradient exploding or vanishing at large ranks. We set :math:`r=8` for memory efficiency. rsLoRA just slightly underperforms against LoRA and QLoRA. The original rsLoRA paper's experiments indicated lower perplexity at higher ranks (e.g., :math:`r = 64`). This lower perplexity and the fact that higher rank LoRA captures more details suggest rsLoRA's benefits are primarily exploited at high ranks.

DoRA Benefits from Two Learning Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DoRA performed worse than the other three LoRA methods. We used the same learning rate for updating the magnitude vector and direction matrix. However, as shown in detailed performance tables, this can lead to sub-optimal performance in some cases due to the gradient scales being different between the two types of updates in DoRA. This leads to DoRA sometimes under-training the magnitude vector in our experiments, which uses the same low learning rate. Thus, DoRA may achieve higher performance if the magnitude vector has its own learning rate that is higher than the low-rank update's learning rate.

LoRA-Tuned Llama 3.1 8B vs. Baseline Models and Gemini Fine-Tuned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compared to SOTA base LLMs, the LoRA-tuned Llama 3.1 8B Instruct models generally show superior performance across most datasets, with NWGI and FNXL being the exceptions. Against another fine-tuned baseline, the Gemini 2.0 FL fine-tuned model, this Gemini model excels in general financial tasks and XBRL data reporting. However, our Llama 3.1 8B Instruct LoRA variants demonstrate stronger average performance in financial analysis and XBRL data analysis tasks.

Angle II: Financial Task LoRA Suitability
-----------------------------------------

.. raw:: html

    <object style="width: 100%" data="../_static/images/p2.svg" type="image/svg+xml"></object>

The above Figure highlights LoRA's varying effectiveness across different financial tasks. A key observation is the contrast in LoRA method improvements between XBRL Analysis tasks and FinanceBench. Although both aim to analyze financial statements, tasks based on XBRL data demonstrate substantial LoRA-induced performance improvements, whereas FinanceBench exhibits minimal gains. This disparity underscores XBRL's superior suitability for financial statement analysis. The standardized semantics and taxonomy inherent in XBRL likely provide a more structured and consistent learning environment for LLMs, facilitating more effective adaptation compared to FinanceBench, which relies on OCR-processed PDF data lacking such rich, standardized metadata. These findings emphasize the crucial role of XBRL in enabling effective LLM integration for financial report analysis.


Angle III: Resource Usage and Performance Trade-offs of LoRA methods
--------------------------------------------------------------------

Detailed tables on fine-tuning costs show the computational expenses of LoRA fine-tuned models. Using four NVIDIA A5000 GPUs, the wall-clock time for fine-tuning ranged from 14.1 hours (QLoRA) to 15.9 hours (DoRA), corresponding to a total of approximately 56.4 to 63.6 GPU hours. At an estimated rate of $0.26 per GPU hour, this translates to a cost of roughly $14.66 to $16.54. This is substantially more cost-effective than fine-tuning services from providers like Google or OpenAI. Illustrations of inference time for fine-tuned models on various datasets indicate that Gemini API generally exhibits lower inference latency and is less sensitive to increasing prompt lengths than local Llama 3.1 8B Instruct inference, even when accounting for network overhead for the API. However, the inference speed of locally deployed Llama models can be significantly enhanced through the use of larger batch sizes.

Angle IV: Practicability of Applying LoRA in Real-world Financial Scenarios
---------------------------------------------------------------------------

Federated LoRA
~~~~~~~~~~~~~~
The sensitive nature of financial data necessitates privacy-preserving techniques like Federated Learning for collaborative training. To explore this, we evaluated Federated LoRA, with results presented in relevant tables. Our experimental setup simulated a four-node environment employing the FedAvg algorithm, where the sentiment analysis dataset was partitioned across these nodes. The performance of this approach was benchmarked against both the base Llama model and standard centralized LoRA fine-tuning. While Federated LoRA did not match the performance levels of centralized LoRA, the results demonstrate a notable improvement compared to the base Llama model.

Catastrophic Forgetting
~~~~~~~~~~~~~~~~~~~~~~~
A major concern with PEFT is that fine-tuning on domain-specific tasks leads to the model forgetting pre-training knowledge. To investigate this, we evaluated eight adapters—covering both sentiment and FiNER tasks and all four LoRA variants—as well as the Llama 3.1 8B Instruct base model on two out-of-domain benchmarks, MMLU and GSM8K. We used a zero-shot, no chain-of-thought setting to isolate stored knowledge. Performance tables focusing on these benchmarks show identical MMLU accuracy across all adapters and the base model, and equal or higher scores on GSM8K. Hence, at the ranks :math:`r` we tested (4 and 8) with :math:`\alpha:r` equal to 8:1 or 4:1, we observe that LoRA does not exhibit catastrophic forgetting. In fact, the slight GSM8K performance improvements hint at cross-domain knowledge transfer—fine-tuning on financial data may improve the model’s numerical reasoning skills.



Case Study
----------

XBRL Tagging
~~~~~~~~

The following sentence were selected from this XBRL filing. https://www.sec.gov/ix?doc=/Archives/edgar/data/0001652044/000165204425000043/goog-20250331.htm#fact-identifier-743

.. list-table::
   :widths: 20 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Input**
     - What is the appropriate XBRL US GAAP tag for "2.0" in the given sentence? Output the US GAAP tag only and nothing else. As of December 31, 2024 and March 31, 2025, equity securities accounted for under the equity method had a carrying value of approximately $2.0 billion in each period.

   * - **Llama 3.1 8B Base output**
     - us-gaap:MajorityEquityInterest


   * - **Llama 3.1 8B Finetuned output**
     - us-gaap:EquityMethodInvestments

   * - **Ground Truth**
     - us-gaap:EquityMethodInvestments

The Base model outputed an incorrect tag as the 2.0 billion in the sentence does not relate to interest. The finetuned model provided the correct response.


XBRL Extraction
~~~~~~~~

Formula Calculation
**********

.. list-table::
   :widths: 20 90
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - **Input**
     - Can you provide the value for Cash Flow Margin from Chevron Corp for the Fiscal Year ending in FY 2019?  Answer with a formula substituted with values.  [XBRL file segment]


   * - **Llama 3.1 8B Base output**
     - (27314000000 / 5536000000) * 100


   * - **Llama 3.1 8B Finetuned output**
     - (27314000000 / 146516000000) * 100

   * - **Ground Truth**
     - (27314000000 / 146516000000) * 100

The base model used the wrong value for the revenue used in the calculation. The finetuned model outputted the correct formula substituted with values.
