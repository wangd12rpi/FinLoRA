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
   * - CFA Level I
     - Analyst Exam
     - 180
     - 90
     - 181
     - Accuracy, F1
   * - CFA Level II
     - Analyst Exam
     - 88
     - 77
     - 1.0k
     - Accuracy, F1
   * - CFA Level III
     - Analyst Exam
     - 80
     - 78
     - 961
     - Accuracy, F1
   * - CPA REG
     - Accountant Exam
     - 124
     - 101
     - 147
     - Accuracy, F1



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