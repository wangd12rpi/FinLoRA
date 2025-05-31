
Quantized LoRA (QLoRA)
~~~~~~~~~~~~~~~~~~~~~~

When fine-tuning, LoRA requires a large amount of GPU memory. To solve this issue, we can use QLoRA.
QLoRA drastically reduces memory usage and lets you fine-tune on a single GPU.

In QLoRA, we quantize the weights of the adapter layers, reducing both parameter count and memory usage. Quantization is a technique that reduces the precision of the weights to reduce the number of bits used to store them. It consists of two parts: Rounding to the nearest integer and truncating to remove the decimal portion of a floating point number. QLoRA specifically uses 4-bit NormalFloat (NF4), an optimal data type for normally distributed weights, quantization. Pre-trained weights are usually normally distributed and centered around 0, which is why NF4 is ideal for quantization.

If we quantize from Float16 to Int4, we can represent 16 different values (bins) because Int4 has 4 bits and :math:`2^{4}=16`. Inputs are usually normalized from -1 to 1. Very close together values, however, will be mapped to the same bin. This means that the precision is lost if we want to convert back to Float16. However, we can use blockwise quantization, where we divide the input range into blocks and quantize each block separately. QLoRA uses a 64 blocksize for better precision.

Since regular quantization relies on the bins being equally probable, QLoRA uses NormalFloat where the bins are weighted by the normal distribution. The spacing between bins is therefore closer together near 0 and further apart further away from 0.

Each block in QLoRA has a quantization constant. QLoRA employs double quantization, where it quantizes the quantization constants themselves to further save space.

The last part of QLoRA is paged optimizers. Paged optimizers reduce GPU memory spikes by switching pages to CPU memory when GPU RAM becomes full when processing long sequences, and the pages are not needed for the current computation of the forward/backward pass.

The forward pass for QLoRA is :math:`\boldsymbol{y} = p_{16}(\boldsymbol{W}_0^{\text{NF4}}) \boldsymbol{x} + \gamma_r \boldsymbol{B} \boldsymbol{A} \boldsymbol{x}`.
