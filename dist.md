# parallelism 
NOTE: will move this to a dedicated readme later
NOTE: copy-pasta'd some sentences and diagrams from hf articles on the subject


## ZeRO-DP
Consider model with 3 layers, La, Lb, Lc.

```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

If we have 3 GPUs ZeRO-DP splits the model like so:

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

This is a form of tensor parallelism (TP) or horizontal slicing since each GPU has a subset of the parameters for a given layer. As we feed mini-batches to each GPU, we broadcast the parameters needed for the computation and then drop them. On GPU0: the x0 mini-batch requires the a0, a1, a2 parameters to do its forward path through the layer, but the GPU0 has only a0. It will get a1 from GPU1 and a2 from GPU2, bringing all the pieces of the model together.

## Improving naive vertical parallelism with pipeline parrallelism
The same model from before could also be paritioned across our GPUs like this:
```
================
| Layer |      |
|   a   | GPU0 |
================
| Layer |      |
|   b   | GPU1 |
================
| Layer |      |
|   c   | GPU2 |
```
This is bad since:
* All but one GPU are idle at any given moment: if 4 GPUs are used, it’s nearly identical to quadrupling the amount of memory of a single GPU, and ignoring the rest of the hardware.
* Overhead in data transfer between devices
* Shared embeddings may need to be copied between GPUs (think llms with weight sharing)

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)

Instead of each GPU operating on a single chunk in a mutually exclusive fashion, we can micro-batch the inputs and each GPU can concurrently work on their own micro-batch in the forward and backwards pass. This is pipeline parallelism (PP). There's still some downtime known as the "bubble" where GPU's are waiting for the results in the fwds and bkwds passes. The goal is to minimize the size of the bubble.

In practice though, PP is fast. 


## Tensor parallelism 
Give each GPU a slice of a some weights. It seeems counter-intuitive at first, but at the end of the day its just basic linear algebra

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel\_gemm.png)

If we split the weight matrix A column-wise across N GPUs and perform matrix multiplications XA\_1 through XA\_n in parallel, then we will end up with N output vectors Y\_1, Y\_2, ..., Y\_n which can be fed into GeLU independently -> Y = [GELU(Y\_1), ..., GELU(Y\_n)]

No inter-GPU communication needs to happen during these operations. 

Not adviseable to use TP across more than one node since it requires a **very fast** network.

From the llama3.1 paper:
>The order of parallelism dimensions, [TP, CP, PP, DP], is optimized for network communication. The innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server. The outermost parallelism may spread across a multi-hop network and should tolerate higher network latency


## multi-dimensional parallelism 
TP, DP, and PP aren't mutually exclusive. By choosing different combinations of parallelism we can achieve higher spedeups. 
![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

# Single-node multi-GPU setup tips

Model fits onto a single GPU:
* DDP - Distributed DP
* ZeRO - may or may not be faster depending on the situation and configuration used

Model doesn’t fit onto a single GPU:
* PP
* ZeRO
* TP

With very fast intra-node connectivity of NVLINK or NVSwitch all three should be mostly on par, without these PP will be faster than TP or ZeRO. The degree of TP may also make a difference. Best to experiment to find the winner on your particular setup.

TP is almost always used within a single node. That is TP size <= gpus per node.

Largest Layer not fitting into a single GPU:
* If not using ZeRO - must use TP, as PP alone won’t be able to fit.
* With ZeRO see the same entry for “Single GPU” above
