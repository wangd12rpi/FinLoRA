Rank-Stabilized LoRA (rsLoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LoRA scales the weight matrix update :math:`\boldsymbol{BA}` by :math:`\frac{\alpha}{r}`, which can cause gradients to explode or diminish as the rank :math:`r` increases. In contrast, rsLoRA uses a scaling factor :math:`\frac{\alpha}{\sqrt{r}}`:

.. math::

   \boldsymbol W'=\boldsymbol W_0+\frac{\alpha}{\sqrt{r}}\boldsymbol B\boldsymbol A.

This scaling results in gradient-scale stability at higher ranks, enabling the rank to be higher to capture more details in long-context tasks like XBRL extraction. rsLoRA also results in lower perplexity—the model assigns higher probabilities to correct words—than LoRA at higher ranks.
