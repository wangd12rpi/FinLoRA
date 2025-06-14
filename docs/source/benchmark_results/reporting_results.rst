=======
Results on Financial Reporting Tasks
=======

.. raw:: html

    <script type="text/javascript">
      function resizeIframe(iframe) {
        iframe.height = iframe.contentWindow.document.body.scrollHeight + "px";
      }
    </script>
    <embed>
        <iframe onload="resizeIframe(this)" src="../_static/tables/reporting_result.html" frameborder="0" width="100%" ></iframe>
    </embed>



Angle I: LoRA Methods' Performance on Financial Datasets
=========================================================

For the FiNER task, standard **LoRA** and **QLoRA** provided the best results. For the more complex FNXL task, **rsLoRA** and **DoRA** performed slightly better, though all Llama variants showed limited gains.

Compared to the closed-source fine-tuned Gemini 2.0 Flash Lite baseline. The Gemini model achieved significantly higher scores across all three datasets, indicating a performance ceiling for the smaller Llama 3.1 8B model on these specific technical tasks, regardless of the LoRA method used.

Angle II: LoRA Suitability for Financial Tasks
================================================

The effectiveness of LoRA fine-tuning on Llama 3.1 8B is highly dependent on the task.

A substantial performance increase was observed on the **FiNER** task, which aligns with the known strengths of fine-tuning for domain-specific pattern and entity recognition.

In contrast, fine-tuning yielded smaller improvement on the **FNXL** and **XBRL Term** tasks. FNXL has a smaller training set while having more classification labels, which might be the cause.
