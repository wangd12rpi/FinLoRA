==================
Financial Data Reporting
==================

Overview
************



.. list-table:: Question/training sets for general financial tasks.
   :widths: auto
   :header-rows: 1

   * - Question sets
     - Type
     - #Train
     - #Test
     - Metrics
     - Source
   * - FiNER-139
     - XBRL Tagging
     -
     -
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/TheFinAI/en-fpb>`__
   * - FNXL
     - XBRL Tagging
     -
     -
     - Accuracy, F1
     - `huggingface <https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification>`__

Tasks Details
************************

XBRL tagging is a crucial step in creating XBRL reports. This process involves tagging numerical entities within texts, such as earnings call transcripts, using US GAAP tags.

Two distinct approaches are employed:

* Regular Approach: The input addresses only one numerical entity at a time, and a list of potential US GAAP tag options is not provided. This approach is designed for simple and quick queries. Since options aren't provided, input processing can be faster. Fine-tuned models perform well after learning the tag taxonomy, but base models perform poorly because they lack knowledge of all the valid tags.

* Batched Approach: The input includes multiple (e.g., four) numerical entities to be tagged simultaneously, and a list of potential US GAAP tag options is provided. Providing the options allows the Large Language Model (LLM) to know which tags are valid choices without needing pre-existing knowledge of the entire taxonomy; it can infer the appropriate tag from its name and context. To improve efficiency and reduce token usage, given the large number of tags, multiple tagging questions are grouped into a single input batch following the list of options.

FiNER-139
--------------------
FiNER includes XBRL Tagging tasks with labels from the 139 most common US GAAP tags.

Regular: 

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Input**
     - What is the appropriate XBRL US GAAP tag for "12,028" in the given sentence? Output the US GAAP tag only and nothing else. "Rent expense for this lease amounted to 12,028 and 12,500 for the six months ended January 31 , 2020 and 2019 , respectively ."
   * - **Output**
     - LeaseAndRentalExpense

Batched: 

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Input**
     - You are XBRL expert.  Here is a list of US GAAP tags options: ,SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage,InterestExpense, ... [omitted for this table] Answer the following 4 independent questions by providing only  4 US GAAP tags answers in the order of the questions. Each answer must be saperated by a comma (,).  Provide nothing else. 1. What is best tag for entity "0.36" in sentence: "On May 27 , 2020 , the Company u2019 s Board of Directors declared a quarterly cash dividend of $ 0.36 per share , which is payable on or before July 21 , 2020 to shareholders of record on July 7 , 2020 .?"2. What is best tag for entity "114" in sentence: "As of March 31 , 2020 , we owned interests in the following assets :  116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ;  90 hotel condominium units at World Quest Resort in Orlando , Florida (  World Quest  ) ; and  17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels .?" 3. What is best tag for entity "two" in sentence: "As of March 31 , 2020 , we owned interests in the following assets :  116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ;  90 hotel condominium units at World Quest Resort in Orlando , Florida (  World Quest  ) ; and  17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels .?" 4. What is best tag for entity "2.8" in sentence: "As of March 31 , 2020 , we owned interests in the following assets :  116 consolidated hotel properties , including 114 directly owned and two owned through a majority - owned investment in a consolidated entity , which represent 24,746 total rooms ( or 24,719 net rooms excluding those attributable to our partner ) ;  90 hotel condominium units at World Quest Resort in Orlando , Florida (  World Quest  ) ; and  17 . 1 % ownership in Open Key with a carrying value of $ 2.8 million . For U.S. federal income tax purposes , we have elected to be treated as a REIT , which imposes limitations related to operating hotels .?" Output US GAAP tags:""
   * - **Output**
     - CommonStockDividendsPerShareDeclared,NumberOfRealEstateProperties,NumberOfRealEstateProperties,EquityMethodInvestments


FNXL
--------------------
FNXL is similar to FiNER, but includes even more US GAAP tags as labels.

Regular: 

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Input**
     - What is the best us gaap tag for entity "187.27" in sentence: "The 2018 ASR Agreement was completed on January 29, 2019, at which time the Company received 117,751 additional shares based on a final weighted average per share purchase price during the repurchase period of $187.27."?
   * - **Output**
     - AcceleratedShareRepurchasesFinalPricePaidPerShare

Batched: 

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Input**
     - {"context": "You are XBRL expert. Choose the best XBRL US GAAP tag for each highlighted entity in the sentences below. Provide only the US GAAP tags, comma-separated, in the order of the sentences and highlighted entity. Provide nothing elseAcceleratedShareRepurchasesFinalPricePaidPerShare, AccountsReceivableFromSecuritization, AccountsReceivableNetCurrent. [omitted for this table] What is the best us gaap tag for entity "6.3" in sentence: "The projected benefit obligation and fair value of plan assets for U.S. pension plans with projected benefit obligations in excess of plan assets was $6.3 billion and $4.7 billion, respectively, as of December31, 2019 and $5.5 billion and $4.1 billion, respectively, as of December31, 2018."?What is the best us gaap tag for entity "124,043" in sentence: "Capitalized software, net of accumulated amortization of $124,043 in 2020 and $104,237 in 2019"?What is the best us gaap tag for entity "1.5" in sentence: "The Company purchased 30 million and 57 million shares under stock repurchase programs in fiscal 2020 and 2019 at a cost of $1.5 billion and $3.8 billion, respectively."?What is the best us gaap tag for entity "651,313" in sentence: "This multi-tenant mortgage loan is interest-only with a principal balance due on maturity, and it is secured by seven properties in six states, totaling approximately 651,313 square feet."??
   * - **Output**
     - DefinedBenefitPlanPensionPlanWithProjectedBenefitObligationInExcessOfPlanAssetsProjectedBenefitObligation,CapitalizedComputerSoftwareAccumulatedAmortization,PaymentsForRepurchaseOfCommonStock,AreaOfRealEstateProperty

Citations
****************
