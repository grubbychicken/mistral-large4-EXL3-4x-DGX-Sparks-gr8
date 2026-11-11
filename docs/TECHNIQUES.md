# Techniques

This page explains how the recipe gets its speed on four DGX Sparks. The method for each step was the same:
measure where the time goes, change one thing, and prove that the replies did not change.

## The problem

Mistral Large 4 has about 1 trillion parameters. For each token it uses about 52 billion of them: 72 layers, and in
each layer 4 experts selected from 128, plus one shared expert. At 3 bits per weight the model is 403 GB. One DGX
Spark has 128 GB of memory, shared between the system and the GPU, so four Sparks are necessary.

A DGX Spark has a fast GPU but ordinary memory speed. For one token the engine must read about 20 GB of weights.
Thus memory speed, not arithmetic, decides the decode speed. Most techniques below reduce the bytes that the GPU
reads, or the waiting time between GPU tasks.

## 1. Compressed weights, decoded inside the kernels

The EXL3 format stores each weight in 3 bits as a path through a code table (a trellis). A usual engine expands the
weights to 16 bits first; that needs five times the memory and five times the reads. Here the GPU kernels decode the
3-bit codes while they multiply. The result is the same number that ExLlamaV3 computes for the same weights.

The recipe does not quantize again. The converter only moves the data into the layout that these kernels read, and
splits it in four. A check then reads each converted tensor back and compares it with the download.

## 2. Tensor parallelism on a ring of cables

Each Spark holds one quarter of each matrix. For each layer the Sparks compute their parts at the same time and then
exchange small results. There are about 300 such exchanges for each token, so each exchange must be very fast.

- The Sparks use RDMA (RoCE v2) directly from GPU-visible memory. A message goes from one GPU to the next without a
  copy by the CPU.
- The cables make a ring. A Spark has no cable to the Spark across the ring; a neighbour relays those messages.
- The attention part of each layer runs as one chain of exchanges in place of many separate ones.

## 3. Speculative decoding with Mistral's draft head

Mistral ships a small draft model (the DSpark head, 4.2 GB) with Mistral Large 4. It reads the hidden states of the
full model and proposes the next tokens. The full model then checks up to three proposed tokens in one step, which
costs little more than one normal step.

- The check computes each row with the same arithmetic as a normal step. Therefore a reply with drafts is identical
  to a reply without drafts. This was verified token by token.
- A draft that the model rejects costs time. The engine reads the draft head's own probability for each proposed
  token and uses a shorter draft when a long one is unlikely to pay.
- A quarter of the draft head runs on each Spark, like the model itself.

## 4. One CUDA graph for each step

A decode step is several thousand small GPU tasks. If the CPU starts them one by one, the start time is larger than
the work. The engine records the full step once as a CUDA graph and replays it. It keeps one graph for each step
shape and each context size range.

## 5. Attention for long contexts

Mistral Large 4 does not let each token look at all earlier tokens. An index selects the important ones (sparse
attention), and the cache holds one compact row for each earlier token in place of a row for each attention head.

- The compact cache needs about 50 MB for each 1,024 tokens on each Spark. That makes 65,536 tokens possible in the
  memory that the weights leave free.
- Each Spark scores one quarter of the index, and the kernels process all attention heads of a Spark together.
- All layers share one work buffer for attention during decoding. That alone freed 4 GB on each Spark.

## 6. Prompt processing (prefill)

A prompt has many tokens at the same time, so the engine can use the tensor cores of the GPU well.

- The prompt is processed in blocks of 2,048 tokens, and two blocks run at the same time.
- In each layer the tokens are grouped by expert. Each group is one matrix multiplication on 16-bit values, with the
  weights decoded from 3 bits in the same kernel.
- Attention runs in parts of 1,024 rows, so its work memory stays small for long prompts.
- The weight records are stored in the order in which the prefill kernels read them.

## 7. Reuse between requests

When a request ends, the engine keeps its cache until the memory is needed. A new request that starts with the same
tokens (the next turn of a chat, or the same system prompt) computes only the new tokens. The state of the draft
head is kept too, so the second turn decodes as fast as the first.

## 8. Several requests in one step

Up to four requests run in the same decode step. The cost of a step grows much less than the number of rows, so the
total speed with four requests is about 1.5 times the speed with one.

## 9. Fast start

The converted weights are one file for each Spark, in the exact layout of the GPU memory. The loader reads it with
eight threads, at about 10 GB per second. The engine frees the file cache first, because on a DGX
Spark cached files use the same memory as the GPU.

## 10. Memory safety

Each Spark has about 4 GiB free while the server runs. The engine allocates all its memory at the start, so the free
memory stays the same for short and long prompts. A guard samples all four Sparks every 5 seconds and stops the
server if one stays under the floor.

## What was tried and not kept

- An 8-bit cache row for attention: the same quality and less memory, but no speed gain. It stays as a build switch.
- Two redesigns of the prefill expert kernels: identical results, but not faster on this GPU. They stay in patch
  0029, off by default.
- Drafts copied from the prompt in place of the draft head: a replay on recorded requests showed no gain.
