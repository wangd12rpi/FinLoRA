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

The data from professional financial examinations (CFA, CPA)—which had some focus on ethics and regulations—shows that base models are largely incapable of handling complex, knowledge-intensive tasks and even more so in complying with regulations. However, LoRA fine-tuning the Llama 3.1 8B model results in a dramatic performance increase.

Unlike in general financial tasks, the **rsLoRA** and **DoRA** variants demonstrate superior results on these reasoning-based exams, outperforming the vanilla LoRA method in several instances. This suggests that for tasks requiring deeper knowledge and higher alignment with regulations, these methods may be more effective.

These results show that LoRA fine-tuning is very well-suited for tasks that depend on expert knowledge and regulation compliance. The process transforms a general-purpose model into a domain expert that complies with regulations.

Angle II: LoRA Suitability for Financial Tasks
================================================

These results show that LoRA fine-tuning is very well-suited for tasks that depend on expert knowledge and regulation compliance. The process transforms a general-purpose model into a domain expert that complies with regulations.