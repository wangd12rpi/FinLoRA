Financial Certification Tasks
=======================




We aim to benchmark our model's performance on professional financial certification exams, including the Chartered Financial Analyst (CFA) exams and Certified Public Accountant (CPA) Regulation exam.

.. list-table:: Question/training sets for financial certification tasks.
   :widths: auto
   :header-rows: 1

   * - Question sets
     - Type
     - #Train
     - #Test
     - Average Prompt Length
     - Metrics
     - Train Data
     - Test Data
   * - CFA Level I
     - Analyst Exam
     - 180
     - 90
     - 181
     - Accuracy, F1
     - Not publicly available due to copyright
     - Not publicly available due to copyright
   * - CFA Level II
     - Analyst Exam
     - 88
     - 77
     - 1.0k
     - Accuracy, F1
     - Not publicly available due to copyright
     - Not publicly available due to copyright
   * - CFA Level III
     - Analyst Exam
     - 80
     - 78
     - 961
     - Accuracy, F1
     - Not publicly available due to copyright
     - Not publicly available due to copyright
   * - CPA REG
     - Accountant Exam
     - 124
     - 101
     - 147
     - Accuracy, F1
     - Not publicly available due to copyright
     - Not publicly available due to copyright



**CFA Level I** (Analyst Exam)
--------------------
CFA Level I exam focuses on ethics, professional standards, and the fundamentals of investment tools and asset valuation.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - Please answer the following question with the exact letter and choice text and no explanations that you choose. Nothing else like an explanation should be in your response.
   * - **Input**
     - Question: Phil Jones, CFA, has just finished researching Alpha One Inc. and is about to issue an unfavorable report on the company. His manager does not want him to state any adverse opinions about Alpha One, as it could adversely affect their firm's relations with the company, which is an important investment banking client. Which of the following actions by the manager most likely violates Standard I (B): Independence and Objectivity?. Choices: A. Putting Alpha One on a restricted list, B. Asking Jones to issue a favorable report, C. Asking Jones to only state facts about the company.
   * - **Output**
     - B. Asking Jones to issue a favorable report

**CFA Level II** (Analyst Exam)
--------------------
CFA Level II exam emphasizes the application of investment tools and concepts with a focus on asset valuation and analysis.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - Look at the following context.
   * - **Input**
     - Glenda Garvey is interning at Samson Securities in the summer before her final year of business school. She is assigned to work with Abigail Tanner, a senior analyst who covers retail companies. Tanner is preparing a report on JC Miracle, a department store chain. Tanner tells Garvey that she has decided to issue a "buy" recommendation for JC Miracle. Tanner asks Garvey to draft the report, emphasizing the positive aspects of the company and downplaying any negative aspects. Garvey is concerned because her own research on JC Miracle has revealed that the company's financial condition is deteriorating. The company's debt level has increased significantly, and its cash flow is barely sufficient to cover its interest payments. Garvey believes that a "sell" recommendation would be more appropriate. When Garvey expresses her concerns to Tanner, Tanner tells her that she has been covering JC Miracle for many years and has a good relationship with the company's management. Tanner also mentions that JC Miracle is an investment banking client of Samson Securities. Garvey drafts the report as instructed, but she is uncomfortable with the situation. Question: According to the CFA Institute Standards of Professional Conduct, what is the most appropriate action for Garvey to take? Choices: A. Draft the report as instructed, but include a footnote expressing her concerns about the company's financial condition. B. Draft the report as instructed, but document her concerns in a separate memo to Tanner. C. Refuse to draft the report and report Tanner's behavior to the compliance department. D. Draft the report as instructed, but express her concerns to Tanner's supervisor.
   * - **Output**
     - D. Draft the report as instructed, but express her concerns to Tanner's supervisor.

**CFA Level III** (Analyst Exam)
--------------------
CFA Level III exam focuses on portfolio management and wealth planning, requiring synthesis of concepts from Levels I and II.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - Look at the following context.
   * - **Input**
     - Maria Harris is a CFA Level 3 candidate and portfolio manager at XYZ Investment Management. She manages portfolios for high-net-worth individuals. One of her clients, John Smith, has the following investor profile: Investment objective: Growth and income Time horizon: Long-term (retirement in 20 years) Risk tolerance: Moderate Constraints: No investments in tobacco or firearms companies Tax situation: High-income tax bracket, prefers tax-efficient investments Liquidity needs: Low (has adequate emergency funds) Legal/regulatory: None Smith's current portfolio allocation is: 60% equities (40% domestic, 20% international) 30% fixed income 5% real estate 5% cash Smith has recently expressed concern about the potential for rising inflation and interest rates. He has asked Harris to adjust his portfolio to protect against these risks while maintaining his long-term investment objectives. Question: Which of the following portfolio adjustments would be most appropriate for Harris to recommend to Smith? Choices: A. Increase allocation to long-term government bonds to 40% of the portfolio. B. Increase allocation to Treasury Inflation-Protected Securities (TIPS) and reduce duration of fixed income holdings. C. Increase allocation to cash to 20% of the portfolio. D. Increase allocation to growth stocks in the technology sector.
   * - **Output**
     - B. Increase allocation to Treasury Inflation-Protected Securities (TIPS) and reduce duration of fixed income holdings.

**CPA REG** (Accountant Exam)
--------------------
The CPA Regulation (REG) exam tests knowledge of federal taxation, business law, and ethics for accounting professionals.

.. list-table::
   :widths: 10 90
   :header-rows: 0
   :stub-columns: 1

   * - **Instruction**
     - Please answer the following question with the exact letter and choice text and no explanations that you choose. Nothing else like an explanation should be in your response.
   * - **Input**
     - Question: A tax return preparer may disclose or use tax return information without the taxpayer's consent to. Choices: A. Facilitate a supplier's or lender's credit evaluation of the taxpayer., B. Accommodate the request of a financial institution that needs to determine the amount of taxpayer's debt to it, to be forgiven., C. Be evaluated by a quality or peer review., D. Solicit additional nontax business..
   * - **Output**
     - C. Be evaluated by a quality or peer review.


Fine-tuning for Financial Certification Tasks
--------------------------------------------------

Due to the copyright restrictions on the certification exam datasets, we cannot provide the exact datasets used for fine-tuning. However, if you have your own collection of certification exam questions, you can use one of the following configurations to fine-tune a model for these tasks:

.. code-block:: bash

   # Vanilla LoRA with 8-bit quantization and rank 8
   python lora/finetune.py cfa_llama_3_1_8b_8bits_r8

   # QLoRA with 4-bit quantization and rank 4
   python lora/finetune.py cfa_llama_3_1_8b_4bits_r4

   # DoRA with 8-bit quantization and rank 8
   python lora/finetune.py cfa_llama_3_1_8b_8bits_r8_dora

   # RSLoRA with 8-bit quantization and rank 8
   python lora/finetune.py cfa_llama_3_1_8b_8bits_r8_rslora

These configurations use different combinations of quantization bits, rank, and LoRA methods:

- **cfa_llama_3_1_8b_8bits_r8**: Vanilla LoRA with 8-bit quantization and rank 8, providing a good balance between performance and efficiency.
- **cfa_llama_3_1_8b_4bits_r4**: QLoRA with 4-bit quantization and rank 4, reducing memory usage at the cost of some precision.
- **cfa_llama_3_1_8b_8bits_r8_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation) with 8-bit quantization and rank 8, which can improve performance by decomposing weights into magnitude and direction components.
- **cfa_llama_3_1_8b_8bits_r8_rslora**: RSLoRA (Rank-Stabilized LoRA) with 8-bit quantization and rank 8, which uses a different scaling factor to improve stability.

The dataset should be formatted in JSONL format with fields for instruction, input, and output, similar to other tasks in this documentation. You would need to replace the dataset path in the configuration with the path to your own certification exam dataset.
