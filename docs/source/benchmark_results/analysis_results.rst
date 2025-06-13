=======
Financial Statement Analysis Tasks Results
=======

.. raw:: html

    <script type="text/javascript">
      function resizeIframe(iframe) {
        iframe.height = iframe.contentWindow.document.body.scrollHeight + "px";
      }
    </script>
    <embed>
        <iframe onload="resizeIframe(this)" src="../_static/tables/analysis_result.html" frameborder="0" width="100%" ></iframe>
    </embed>

Angle I: LoRA Methods' Performance on Financial Datasets
=========================================================

On financial statement analysis tasks, fine-tuning the Llama 3.1 8B model provides a substantial performance uplift, improving its capabilities on complex formula construction and calculation tasks where base models fail.

Different LoRA methods excel at different tasks. While standard **LoRA** is effective, **DoRA** and **rsLoRA** achieve the highest scores on the formula-based tasks. When compared to the fine-tuned Gemini baseline, the top-performing Llama 3.1 8B variants (using DoRA and rsLoRA) demonstrate superior performance on specific formula construction and calculation. However, the Gemini model shows stronger results on the Financial Math dataset.

Angle II: LoRA Suitability for Financial Tasks
================================================

LoRA fine-tuning is highly effective for teaching models structured, multi-step financial analysis on XBRL format. The most significant performance gains are seen in **Formula Construction** and **Formula Calculation**, where fine-tuning improves the model's capabilities to high proficiency.

The benefit is less pronounced for the broader **FinanceBench** and **Financial Math** datasets. While scores improve, the gains are more modest. This suggests that while LoRA methods are excellent for instilling specific, XBRL-based structured format, they are less effective at enhancing the model's general, open-ended mathematical and financial abilities.
