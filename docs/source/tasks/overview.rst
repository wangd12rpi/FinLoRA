Overview
===================

Task Categories Summary
******************

.. list-table:: Financial Task Categories Overview
   :widths: 20 15 15 25 25
   :header-rows: 1
   :align: left

   * - **Task Category**
     - **Train Size**
     - **Test Size**
     - **Key Applications**
     - **Primary Use Cases**
   * - General Financial Tasks
     - 122.9k
     - 31.7k
     - Market Analysis, Content Processing
     - Sentiment analysis, entity recognition, headline classification
   * - Certification Tasks
     - 472
     - 346
     - Professional Certification
     - CFA Level I/II/III, CPA REG exam preparation
   * - Financial Reporting Tasks
     - 15.9k
     - 8.3k
     - Regulatory Compliance
     - XBRL tagging, terminology, financial reporting standards
   * - Financial Analysis Tasks
     - 27.9k
     - 7.3k
     - Statement Analysis
     - XBRL extraction, formula calculation, financial mathematics

Why These Financial Tasks Matter
*****************

We have selected four main categories of financial tasks, each representing a critical area of financial application.

**General Financial Tasks** are commonly used to benchmark financial language models. These tasks are essential for financial companies to assist with trading, market analysis, and content processing. While they don't require specialized financial knowledge, financial models typically outperform general-purpose models on these tasks.

**Certification Tasks** explore professional-level competency by incorporating questions from financial analyst and accounting mock exams. This includes CFA (Chartered Financial Analyst) exams, which test investment management expertise, and CPA (Certified Public Accountant) exams, which evaluate accounting and regulatory knowledge.

**Financial Reporting Tasks** address highly specialized professional work of creating financial reporting that met the requirements from regulatory agencies, focusing on XBRL (eXtensible Business Reporting Language) - the standardized format for business reporting. These tasks require specific domain knowledge and technical expertise in financial reporting standards.

**Financial Analysis Tasks** involve interpreting financial reports. This includes reading and analyzing financial statements in both PDF and XBRL formats, requiring advanced analytical capabilities and financial reasoning skills.

General Financial Tasks
-----------------------

**Sentiment Analysis** drives market intelligence and risk management:

- Financial sentiment directly influences market movements and investor behavior
- Real-time sentiment analysis enables automated trading and risk assessment

**Headline Analysis** enables rapid information processing:

- Headlines often move markets before full articles are read
- Automated filtering processes thousands of financial headlines efficiently

**Named Entity Recognition** structures unstructured financial data:

- Maps relationships between financial entities for compliance tracking
- Extracts structured data from financial texts for research automation

Certification Tasks
-------------------

**CFA (Chartered Financial Analyst)** represents the gold standard in investment management:

- Tests comprehensive financial analysis capabilities required for financial roles

**CPA REG (Certified Public Accountant - Regulation)** ensures regulatory compliance:

- Covers essential taxation rules and audit requirements
- Protects public financial interests through professional liability standards

Financial Reporting Tasks
-------------------------

**XBRL Terminology** enables standardized financial reporting:

- Understanding SEC-mandated format for public company filings ensures regulatory compliance

**XBRL Tagging** (FiNER and FNXL) ensures accurate financial presentation:

- Proper tagging prevents regulatory violations and improves data quality
- Enables comparative analysis and automated processing across companies

Financial Analysis Tasks
------------------------

**XBRL Extraction Tasks** automate financial data processing:

- Extracts specific data for company financial review
- Enables regulatory monitoring and performance tracking at scale

**Formula Construction and Calculation** supports financial modeling:

- Ensures precise ratio analysis and performance measurement

**FinanceBench** provides real-world analytical capabilities:

- Represent actual analyst work with comprehensive financial report analysis

Why Financial AI Tasks Are Critical
***********************

The financial industry demands exceptional accuracy and reliability because errors can result in regulatory violations, substantial financial losses, and market instability. Financial professionals face significant legal liability, and the entire financial system depends on public confidence in accuracy and integrity.

FinLoRA's comprehensive task coverage ensures AI systems can handle critical financial functions while maintaining the rigorous standards required by the financial industry.

Related Documentation
*******************

For detailed information about each task category:

- :doc:`general_financial_tasks` - Sentiment analysis, headline classification, and NER
- :doc:`certification_tasks` - CFA and CPA professional certification tasks
- :doc:`xbrl_reporting_tasks` - XBRL terminology and tagging tasks
- :doc:`xbrl_analysis_tasks` - XBRL extraction, formula, and analysis tasks
- :doc:`dataset_processing` - Data preparation and processing methods

For implementation details:

- :doc:`../tutorials/setup` - Environment setup and configuration
- :doc:`../tutorials/finetune` - Fine-tuning models for financial tasks
- :doc:`../tutorials/eval` - Evaluation methods and metrics
- :doc:`../benchmark_results/overview` - Performance results across all tasks