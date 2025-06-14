=======
Results on Financial Certification Tasks
=======

.. raw:: html

    <script type="text/javascript">
      function resizeIframe(iframe) {
        iframe.height = iframe.contentWindow.document.body.scrollHeight + "px";
      }
    </script>
    <embed>
        <iframe onload="resizeIframe(this)" src="../_static/tables/certificate_result.html" frameborder="0" width="100%" ></iframe>
    </embed>


Angle I: LoRA Methods' Performance on Financial Datasets
=========================================================

The data from professional financial examinations (CFA, CPA) shows that base models are largely incapable of handling these complex, knowledge-intensive tasks, with most failing to achieve passing scores. However, LoRA fine-tuning the Llama 3.1 8B model results in a dramatic performance increase, achieving passing grades across all exams.

Different from general financial tasks, the **rsLoRA** and **DoRA** variants demonstrate superior results on these reasoning-based exams, outperforming the standard LoRA method in several instances. This suggests that for tasks requiring deeper knowledge, these methods may be more effective.

The fine-tuned Gemini 2.0 FL model showed unexpectedly poor performance, scoring lower than even the base GPT-4o model. One potential reason is Gemini's proprietary fine-tuning methods does not work well on smaller training size.

Angle II: LoRA Suitability for Financial Tasks
================================================

These results show that LoRA fine-tuning is very well-suited for tasks that depend on expert knowledge. The process effectively transforms a general-purpose model into a domain expert that can achieve professional-level performance.
