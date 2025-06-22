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
   * - Financial Math
     - Math
     - 800
     - 200
     - 116
     - Accuracy
     - `github <https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/formulas_with_explanations_with_questions_with_gt.xlsx>`__
   * - FinanceBench
     - Math
     - 86
     - 43
     - 983
     - BERTScore
     - `github <https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/financebench.xlsx>`__

