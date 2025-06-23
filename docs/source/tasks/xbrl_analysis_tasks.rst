==================
Financial Statement Analysis
==================

.. list-table:: Question/training sets for XBRL analysis tasks.
   :widths: auto
   :header-rows: 1

   * - Question sets
     - Type
     - #Train
     - #Test
     - Average Prompt Length
     - Metrics
     - Source
   * - Financial Math
     - Math
     - 800
     - 200
     - 116
     - Accuracy
     - `github <https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/formulas_with_explanations_with_questions_with_gt.xlsx>`__
   * - Tag Extraction
     - XBRL Extraction
     - 10.1k
     - 2.9k
     - 3.8k
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
   * - Value Extraction
     - XBRL Extraction
     - 10.1k
     - 2.5k
     - 3.8k
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
   * - Formula Construction
     - XBRL Extraction
     - 3.4k
     - 835
     - 3.8k
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
   * - Formula Calculation
     - XBRL Extraction
     - 3.4k
     - 835
     - 3.8k
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
   * - FinanceBench
     - Math
     - 86
     - 43
     - 983
     - BERTScore
     - `github <https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/financebench.xlsx>`__


Financial Statement analysis involves extracting and interpreting financial data from XBRL-formatted documents. This process enables users to analyze financial statements, extract specific values, construct formulas, and perform calculations based on the extracted data.

The XBRL analysis tasks can be categorized into several types:

* **Financial Math**: As a starting point of financial analysis, financial math involves solving financial mathematics problems. This requires mathematical knowledge and financial domain expertise.

* XBRL Analysis:  Analyze financial reports in XBRL format
    * **Tag Extraction**: Identifying and extracting specific XBRL tags from financial documents. This task requires understanding the taxonomy and structure of XBRL documents to locate the correct tags.

    * **Value Extraction**: Retrieving the numerical values associated with specific XBRL tags. This involves not only finding the tag but also extracting the corresponding value.

    * **Formula Construction**: Creating formulas that define relationships between different XBRL tags. This requires understanding the financial concepts represented by the tags and how they relate to each other.

    * **Formula Calculation**: Applying constructed formulas to calculate financial metrics based on extracted values. This involves performing mathematical operations on the extracted data.

* **FinanceBench**: Analyzing and interpreting financial reports in PDF format after OCR.


Financial Math
--------------------
Financial Math involves solving complex financial mathematics problems using data from XBRL documents.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a financial mathematics expert. Your task is to solve the financial problem using the provided data. Provide a clear and concise answer.
   * - **Input**
     - A company reported the following financial data:
       - Revenue: $10,000,000
       - Cost of Goods Sold: $6,000,000
       - Operating Expenses: $2,000,000
       - Interest Expense: $500,000
       - Tax Rate: 25%

       Calculate the company's Net Income and Return on Sales (Net Income / Revenue).
   * - **Output**
     - Net Income: $1,125,000
       Return on Sales: 11.25%

Tag Extraction
--------------------
Tag Extraction involves identifying the appropriate XBRL tag for a specific financial item mentioned in a question, based on the provided XBRL document.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a knowledgeable XBRL assistant. Your task is to analyze the XBRL context and provide an accurate and very concise answer to the question. DO NOT output xml, code, explanation or create new question.
   * - **Input**
     - XML File: <us-gaap:CashAndCashEquivalentsAtCarryingValue>2757000000</>\n<us-gaap:AccountsReceivableNetCurrent>3317000000</>\n<us-gaap:InventoryNet>24886000000</>\n<us-gaap:AssetsCurrent>32471000000</>\n<us-gaap:PropertyPlantAndEquipmentNet>25631000000</>\n<us-gaap:Assets>76445000000</>\n<us-gaap:LiabilitiesCurrent>23110000000</>\n<us-gaap:Liabilities>74883000000</>\n<us-gaap:StockholdersEquity>1562000000</>\n<us-gaap:LiabilitiesAndStockholdersEquity>76445000000</>\n

       Question: What is the US GAAP XBRL tag for Total Equity as reported by Home Depot Inc for the Fiscal Year ending in FY 2023?
   * - **Output**
     - us-gaap:StockholdersEquity

Value Extraction
--------------------
Value Extraction involves retrieving the numerical value associated with a specific XBRL tag from the provided XBRL document.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a knowledgeable XBRL assistant. Your task is to analyze the XBRL context and provide an accurate and very concise answer to the question. DO NOT output xml, code, explanation or create new question.
   * - **Input**
     - XML File: <us-gaap:CashAndCashEquivalentsAtCarryingValue>2757000000</>\n<us-gaap:AccountsReceivableNetCurrent>3317000000</>\n<us-gaap:InventoryNet>24886000000</>\n<us-gaap:AssetsCurrent>32471000000</>\n<us-gaap:PropertyPlantAndEquipmentNet>25631000000</>\n<us-gaap:Assets>76445000000</>\n<us-gaap:LiabilitiesCurrent>23110000000</>\n<us-gaap:Liabilities>74883000000</>\n<us-gaap:StockholdersEquity>1562000000</>\n<us-gaap:LiabilitiesAndStockholdersEquity>76445000000</>\n

       Question: What is the value of Inventory for Home Depot Inc for the Fiscal Year ending in FY 2023?
   * - **Output**
     - 24,886,000,000

Formula Construction
--------------------
Formula Construction involves creating a formula that defines the relationship between different XBRL tags based on financial principles.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a knowledgeable XBRL assistant. Your task is to analyze the XBRL context and provide an accurate formula based on the question. DO NOT output xml, code, explanation or create new question.
   * - **Input**
     - XML File: <us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax>383285000000</>\n<us-gaap:CostOfGoodsAndServicesSold>214137000000</>\n<us-gaap:GrossProfit>169148000000</>\n<us-gaap:ResearchAndDevelopmentExpense>29915000000</>\n<us-gaap:SellingGeneralAndAdministrativeExpense>24932000000</>\n<us-gaap:OperatingExpenses>54847000000</>\n<us-gaap:OperatingIncomeLoss>114301000000</>\n<us-gaap:NonoperatingIncomeExpense>-565000000</>\n<us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest>113736000000</>\n<us-gaap:IncomeTaxExpenseBenefit>16741000000</>\n<us-gaap:NetIncomeLoss>96995000000</>\n

       Question: Construct a formula for calculating Gross Profit Margin using the appropriate XBRL tags.
   * - **Output**
     - us-gaap:GrossProfit / us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax

Formula Calculation
--------------------
Formula Calculation involves applying a formula to calculate a financial metric based on the values associated with XBRL tags.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a knowledgeable XBRL assistant. Your task is to analyze the XBRL context and calculate the requested financial metric. DO NOT output xml, code, explanation or create new question.
   * - **Input**
     - XML File: <us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax>383285000000</>\n<us-gaap:CostOfGoodsAndServicesSold>214137000000</>\n<us-gaap:GrossProfit>169148000000</>\n<us-gaap:ResearchAndDevelopmentExpense>29915000000</>\n<us-gaap:SellingGeneralAndAdministrativeExpense>24932000000</>\n<us-gaap:OperatingExpenses>54847000000</>\n<us-gaap:OperatingIncomeLoss>114301000000</>\n<us-gaap:NonoperatingIncomeExpense>-565000000</>\n<us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest>113736000000</>\n<us-gaap:IncomeTaxExpenseBenefit>16741000000</>\n<us-gaap:NetIncomeLoss>96995000000</>\n

       Question: Calculate the Gross Profit Margin for Apple Inc for the Fiscal Year ending in FY 2023.
   * - **Output**
     - 0.4413 or 44.13%


FinanceBench
--------------------
FinanceBench involves analyzing and interpreting financial benchmarks and metrics from XBRL data.

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - You are a financial analyst. Your task is to analyze the company's performance against industry benchmarks using the provided data. Provide a concise analysis.
   * - **Input**
     - Company: Apple Inc.
       Industry: Technology Hardware

       Company Metrics:
       - Gross Profit Margin: 44.13%
       - Operating Margin: 29.82%
       - Net Profit Margin: 25.31%
       - Return on Assets: 20.35%
       - Return on Equity: 160.09%

       Industry Benchmarks:
       - Gross Profit Margin: 35.20%
       - Operating Margin: 18.50%
       - Net Profit Margin: 15.70%
       - Return on Assets: 12.40%
       - Return on Equity: 22.30%

       Analyze the company's performance relative to industry benchmarks.
   * - **Output**
     - Apple Inc. outperforms industry benchmarks across all metrics. The company's gross profit margin (44.13% vs 35.20%), operating margin (29.82% vs 18.50%), and net profit margin (25.31% vs 15.70%) are significantly higher than industry averages, indicating superior operational efficiency. Return on assets (20.35% vs 12.40%) shows effective asset utilization, while return on equity (160.09% vs 22.30%) demonstrates exceptional shareholder value creation, though this extremely high figure may reflect significant share buybacks or debt leverage.



