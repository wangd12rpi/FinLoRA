
Low‑Rank Adaptation Methods for Large Language Models
====================================================

.. contents::
   :local:
   :depth: 2


1  Introduction
===============

Updating *every* parameter of a multi‑billion‑weight Transformer is usually
impossible on commodity hardware.  *Low‑Rank Adaptation* (**LoRA**)
(**STILL NEED TO ADD CITATION**) and its 4‑bit quantised variant
*Quantised LoRA* (**QLoRA**) (**STILL NEED TO ADD CITATION**) replace full
fine‑tuning with a compact set of trainable matrices, cutting memory and
wall‑clock time by orders of magnitude.  When multiple parties must
collaborate without moving sensitive data, *Federated LoRA*
(**STILL NEED TO ADD CITATION**) provides an adapter‑only aggregation
protocol that preserves privacy.

This note gives a rigorous but implementation‑oriented treatment of

* **LoRA**   – rank‑*r* adapters in full‑precision backbones;
* **QLoRA**  – 4‑bit backbones with mixed‑precision adapters;
* **Federated LoRA** – cross‑organisation training via adapter averaging.

Where useful we assume a single Transformer projection  
:math:`\mathbf W\in\mathbb R^{d\times d}` and an activation
:math:`\mathbf x\in\mathbb R^{d}`.


2  Low‑Rank Adaptation (LoRA)
=============================

2.1  Formal definition
----------------------

The full fine‑tune update :math:`\Delta\mathbf W` is replaced by a rank‑*r*
product

.. math::
   \Delta\mathbf W = \mathbf B\mathbf A,\qquad
   \mathbf A\in\mathbb R^{r\times d},\;
   \mathbf B\in\mathbb R^{d\times r},\quad r\ll d.

The modified projection is therefore

.. math::
   \mathbf y = \bigl(\mathbf W + \mathbf B\mathbf A\bigr)\mathbf x
            = \mathbf W\mathbf x + \mathbf B\bigl(\mathbf A\mathbf x\bigr).

During inference one may *merge* once

.. math::
   \mathbf W' = \mathbf W + \mathbf B\mathbf A,

so that runtime latency and memory footprint are identical to the base model.

2.2  Parameter and memory complexity
------------------------------------

* **Trainable parameters per projection**

  .. math:: N_{\text{train}} = 2dr.

* **Relative fraction of full fine‑tune**

  .. math:: \rho = \frac{2r}{d}.

  With :math:`d=4096` and :math:`r=8` one has :math:`\rho\approx0.39\%`.

* **Activation memory** is unchanged; adapter weights add
  :math:`\mathcal O(dr)` bytes to the checkpoint (typically a few MB for the
  entire model).

2.3  Adapter placement
----------------------

LoRA is agnostic to layer type; any subset of dense projections can be
adapted.  Empirically, adding adapters to the attention projections
(Q, K, V, O) delivers the best *accuracy‑per‑parameter*, but MLP or embedding
layers can be included when domain shift is severe.

2.4  Rank‑selection heuristics
------------------------------

+---------------+----------------------------------------------+
| **Rank *r***  | **Typical usage**                            |
+===============+==============================================+
| 4 – 8         | < 7 B backbones, light classification tasks  |
+---------------+----------------------------------------------+
| 8 – 16        | General instruction tuning up to 30 B        |
+---------------+----------------------------------------------+
| 16 – 64       | Reasoning‑heavy tasks or ≥ 70 B backbones     |
+---------------+----------------------------------------------+

Higher rank ⇒ more parameters ⇒ better adaptation (with diminishing returns).


3  Quantised LoRA (QLoRA)
=========================

QLoRA stores the *frozen* backbone weights in 4‑bit NormalFloat (NF4) while
keeping adapters in 16‑bit (or bfloat16).  This allows, for example, a 65 B
model to be fine‑tuned on an A100‑40 GB.

3.1  Key innovations
--------------------

#. **NF4 codebook** – learned 4‑bit codebook optimal for approximately
   Gaussian weight distributions.  **STILL NEED TO ADD CITATION**
#. **Double quantisation** – 8‑bit secondary quantisation of block‑wise scale
   factors (overhead ≈ 0.13 bit / param).
#. **Paged optimisers** – GPU↔CPU paging of optimiser states to keep VRAM
   constant with sequence length.

3.2  Memory profile (training, batch = 1)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Model
     - Full FT (16‑bit)
     - LoRA (16‑bit base)
     - **QLoRA (4‑bit base)**
   * - 7 B
     - 14 GB
     - 9 GB
     - **5 GB**
   * - 13 B
     - 28 GB
     - 16 GB
     - **10 GB**
   * - 33 B
     - 70 GB
     - 40 GB
     - **21 GB**
   * - 65 B
     - 140 GB
     - 80 GB
     - **41 GB**

3.3  Implementation checklist
-----------------------------

* Quantise weights in 64‑value blocks; scale factors in 256‑value blocks.  
* Store adapters in bfloat16; accumulate gradients in float32.  
* After training, *merge* adapters into the 4‑bit backbone to remove
  de‑quantisation overhead at inference.


4  Federated LoRA
=================

Federated LoRA trains adapters across independent data silos and aggregates
only the low‑rank updates, reducing both communication and privacy risk.

4.1  Training protocol (FedAvg)
-------------------------------

.. code-block:: text

   for round t = 1 … T:
       each client k:
           pull global backbone W_t
           train local (A_k, B_k) on D_k
           upload ΔW_k = B_k A_k
       server:
           ΔW̄ ← (1/K) Σ_k ΔW_k
           W_{t+1} ← W_t + ΔW̄
           broadcast W_{t+1}

With :math:`r=8`, a 70 B model produces ~10 MB per client per round.

4.2  Privacy enhancements
-------------------------

* **Secure aggregation** (secret sharing or homomorphic encryption) hides
  individual updates.  
* **Differential privacy** adds Gaussian noise to each :math:`ΔW_k`.  
* **Content‑addressed storage** (e.g.\ IPFS) provides tamper evidence.

4.3  Example use case – inter‑bank fraud detection
--------------------------------------------------

Three banks train domain‑specific adapters on their private logs.  Federated
LoRA lifts macro‑AUROC by 4–6 % over the best single‑bank model while
revealing no raw data.


5  Method‑selection matrix
==========================

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20

   * - Deployment constraint
     - **LoRA**
     - **QLoRA**
     - **Federated LoRA**
   * - GPU ≥ 80 GB
     - ✓ (highest accuracy)
     - (optional)
     - (optional)
   * - GPU 24 – 48 GB
     - Models ≤ 13 B
     - **Models ≤ 33 B**
     - (optional)
   * - Consumer GPU ≤ 16 GB
     - Models ≤ 7 B
     - **Models ≤ 13 B**
     - —
   * - Multi‑organisation, privacy
     - —
     - —
     - **✓**
   * - Lowest inference latency
     - **✓** (merge)
     - **✓** (merge)
     - ✓ (after merge)


6  Conclusion
=============

LoRA reduces trainable parameters to < 1 % with negligible loss in
expressiveness.  QLoRA extends the idea to 4‑bit backbones, enabling
single‑GPU fine‑tuning of models that formerly required a cluster.
Federated LoRA adds a privacy‑preserving layer for cross‑organisation
collaboration.  Together these techniques form a practical toolkit for
cost‑effective adaptation of large language models.

