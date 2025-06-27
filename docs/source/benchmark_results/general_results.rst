=======
Results on General Financial Tasks
=======

Detailed Results
-----

.. raw:: html

    <script type="text/javascript">
      function resizeIframe(iframe) {
        iframe.height = iframe.contentWindow.document.body.scrollHeight + "px";
      }
    </script>
    <embed>
        <iframe onload="resizeIframe(this)" src="../_static/tables/general_result.html" frameborder="0" width="100%" ></iframe>
    </embed>


Results Analysis
----



Angle I: LoRA Methods' Performance on Financial Datasets
=========================================================

Among the tested LoRA methods, the vanilla LoRA approach [LoRA]_ proved to be the most effective and reliable for general financial tasks. It consistently achieved the highest performance. Other variants like **DoRA** [DoRA]_ and **rsLoRA** [RSLoRA]_ showed performance degradation on more complex tasks (e.g., NWGI [NWGI]_), making them less dependable for broad financial applications.

When benchmarked against SOTA models, LoRA results consistently surpasses specialized models like BloombergGPT [BloombergGPT]_ which are pre-trained on financial data.


Angle II: LoRA Suitability for Financial Tasks
================================================

The largest performance gains were observed in **pattern recognition and classification** tasks, such as Named Entity Recognition (NER) [NER]_, sentiment analysis (FiQA SA) [FiQA]_, and news classification (Headline) [Headline]_. For these tasks, fine-tuning allows the model to learn the specific vocabulary, entities, and sentiment of the financial domain, leading to significant improvements.

Conversely, NWGI [NWGI]_ saw the most modest gains. It might be due to NWGI uses five sentiment labels instead of three, causing the model unable to distinguish between more nuanced differences in sentiment.
