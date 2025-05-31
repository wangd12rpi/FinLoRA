
LoRA with Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the finance sector, multiple banks may want to work together on a model to predict credit risk and whether a borrower will default on a loan. Each bank may have a different dataset, but they cannot share their data due to compliance reasons and privacy concerns. Federated learning solves this issue by fine-tuning a model on local data and aggregating updates during backpropagation to a centralized model via a server.

Differentially Private Low-Rank Adaptation (DP-LoRA) [5]_ is a method to use federated learning with LoRA.

DP-LoRA first uses a server to send the current global LoRA weights (the A and B matrices from earlier) to all clients.

Every client does the following: 1) Gets a minibatch of its private data 2) Computes the gradient for only its local A and B weights clipped with an :math:`\ell_2` norm (square root of the sum of the squares of elements in the vector) 3) Adds Gaussian noise to the gradients 4) Updates the A and B LoRA matrices 5) Sends the updated A and B matrices to the server.

By adding noise, DP-LoRA prevents the centralized model from inferring the private data later on. This would allow the banks in the credit risk example to work on a model together.

As in normal federated learning, the server then aggregates the weights from all clients in a weighted average and sends the updated weights to all clients.