Sample Outputs
----------

XBRL Tagging
~~~~~~~~

The following sentence were selected from this XBRL filing. https://www.sec.gov/ix?doc=/Archives/edgar/data/0001652044/000165204425000043/goog-20250331.htm#fact-identifier-743

.. list-table::
   :widths: 31 23 23 23
   :header-rows: 1
   :stub-columns: 0
   :align: left

   * - **Input**
     - **Llama 3.1 8B Base**
     - **Llama 3.1 8B Fine-tuned**
     - **Ground Truth**
   * - What is the appropriate XBRL US GAAP tag for "2.0" in the given sentence? As of December 31, 2024 and March 31, 2025, equity securities accounted for under the equity method had a carrying value of approximately $2.0 billion in each period.
     - us-gaap: MajorityEquityInterest
     - us-gaap: EquityMethodInvestments
     - us-gaap: EquityMethodInvestments

The base model mistags a $2.0 billion value because it relies on superficial keyword matches (e.g., "equity," "carrying value,") and applies the generic tag
us-gaap:MajorityEquityInterest, ignoring the context "under the equity method."

XBRL Extraction
~~~~~~~~

Formula Calculation
**********

.. list-table::
   :widths: 31 23 23 23
   :header-rows: 1
   :align: left

   * - **Input**
     - **Llama 3.1 8B Base**
     - **Llama 3.1 8B Fine-tuned**
     - **Ground Truth**
   * - Can you provide the value for Cash Flow Margin from Chevron Corp for the Fiscal Year ending in FY 2019?  Answer with a formula substituted with values.  [XBRL file segment]
     - (27314000000 / 5536000000) * 100
     - (27314000000 / 146516000000) * 100
     - (27314000000 / 146516000000) * 100

The base model incorrectly selects a tag referencing the value 1,209,000,000 by matching the keyword
"Equity" and also ignores the decimals="-6" attribute.