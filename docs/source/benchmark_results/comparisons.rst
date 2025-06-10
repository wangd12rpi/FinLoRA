

Output Comparison
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
