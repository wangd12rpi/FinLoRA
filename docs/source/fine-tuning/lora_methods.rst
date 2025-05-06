Low‑Rank Adaptation Methods for Large Language Models
=======================================================

.. contents::
   :local:
   :depth: 2


1. What is LoRA?
----------------
LoRA is a method to efficiently update the parameters  
of pre‑trained language models when fine‑tuning on new tasks.


2. Foundations of LoRA
----------------------

2.1 Ranks
~~~~~~~~~
Rank is the number of linearly independent rows or columns  
in a matrix. Linearly independent columns, for example, are  
columns whose values can't be computed by an addition of  
previous columns multiplied by an integer.

::

    W = [1  7  2  8  5
         2 10  4 12 10
         3 15 12 18 27
         4 12 16 16 36]   ---> Dimensions : 4 x 5
                               rows         columns

In the above matrix, there are 2 linearly independent columns,  
so the rank is 2.

• Column 1 has no previous rows, so it is linearly independent.  
• Column 2 can't be computed as a multiple of column 1, so  
  it is linearly independent.  
• Columns 3‑5 are linearly dependent.  
    • C3 = 2C1 + 0C2  
    • C4 = 1C1 + 1C2  
    • C5 = 1C1 + 2C2  

If we convert the formulas to vectors, we can represent them  
as the following :

.. code-block:: text

       [1]
       [0]
       [2]
       [1]  C1  +  [0]
                   [1]
                   [0]
                   [2]  C2   or equivalently

.. code-block:: text

       [1 0]
       [0 1]
       [2 0]
       [1 2]   or   [1 0 2 1 1
                     0 1 0 1 2]

If we take the matrix multiplication of the linearly independent  
columns and the representation above, it equals the original  
matrix **W**.

cont. on back


Low‑rank decomposition example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    [1  7  2  8  5
     2 10  4 12 10
     3 15 12 18 27
     4 12 16 16 36]  ->  W  =  [1  7
                                 2 10
                                 3 15
                                 4 12] [1 0 2 1 1
                                         0 1 0 1 2]

    Dimensions(W)   = d x k = 4 x 5
    Dimensions(A)   = d x r = 4 x 2   r = rank (rank = 2)
    Dimensions(B)   = r x k = 2 x 5
    Dimensions(A*B) = (d x r) * (r x k) = d x k = Dimensions(W)

    Parameters(W)   = 4 x 5 = 20
    Parameters(A)   = 4 x 2 =  8
    Parameters(B)   = 2 x 5 = 10
    Parameters(A+B) = 8 + 10 = 18

∴ Less parameters are stored if we use the representation  
  of the **A** and **B** matrices.

IF r << min{d,k}, this would be used due to  
having to store less parameters. This is called *low‑rank*.

In the example, 2 << min{4,5} = 2 << 4.


2.2 1 Fine‑tuning Without Adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Say we have a pre‑trained model **M** with **500 million**  
parameters. M has the below architecture.

*Insert Model M Architecture Picture*

Say we pre‑tuned M with two tasks. Task 1 is **Masked Language Modeling (MLM)**, where we mask some words in a sentence, and the task is to predict the sentence with the masked tokens filled in. Task 2 is **Next Sentence Predicting (NSP)**, where the task is to predict if, given 2 sentences, whether sentence A comes before sentence B.

Say we want to fine‑tune pre‑trained model M on a new task **Named Entity Recognition (NER)**, where the task is to annotate one entity (location/person/organization) per sentence in a financial task.

When we fine‑tune the model, all parameters are updated during back‑propagation. Back‑propagation is where we compare the error (difference between the predicted output and the actual output) and send the error backwards through the model, computing the gradient of error with respect to each weight.

If we want to fine‑tune model M on another task **Financial Phrase Bank (FPB)**, where the task is to annotate sentences from financial news and reports with sentiment, we still need to update all 500 million parameters. This is costly and can lead to over‑fitting and the model forgetting pre‑training tasks.

*Insert Back‑propagation Picture*


2.2.2 Fine‑tuning With Adapters (Parameter Efficient Finetuning — PEFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Say instead, when we want to fine‑tune the pre‑trained model M we use **Parameter Efficient Finetuning (PEFT)**, where we add two adapter layers per transformer layer. The architecture of M now looks like the following.

*Insert Model M Architecture With Adapters Picture*

Now, when we fine‑tune M on NER, only the parameters of the adapter layer are updated, but the other weights/parameters are frozen, so during back‑propagation, the gradients of error pass through them, but those weights/parameters aren't updated. While we do have to replace the adapters and store the updated params separately for FPB, the number of parameters is now much smaller.


3 Low‑Rank Adaptation (LoRA)
-----------------------------
Say instead of PEFT, we fine‑tune with **Low‑Rank Adaptation**. 
