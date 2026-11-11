# Patches to gr8

`start.sh` takes gr8 at commit `069ec213fc72b5ed18ac4f6bf1c1a296dae1fb16` and applies these patches in order. They add
Mistral Large 4 as a new model to the engine. Each patch was accepted only when the engine's replies stayed
identical to reference replies, or (for the first patches) when the output agreed with a reference implementation
in PyTorch.

| Patch | What it adds |
|---|---|
| 0001 | The EXL3 expert format of this quant (code table "mul1") in the decode kernel for experts. |
| 0002 | The shape of Mistral Large 4: 72 layers, its position encoding, and which layers share an attention index. |
| 0003 | Kernels for the dense 3-bit matrices and for the 16-bit matrices that the quant keeps. |
| 0004 | The attention block and the expert router of Mistral Large 4 in the decode step. |
| 0005 | The engine on four Sparks in a ring, and an output mode to compare its probabilities with a reference. |
| 0006 | Prompt processing (prefill): 16-bit tensor-core kernels that decode the 3-bit weights. |
| 0007 | Tensor-core attention for prefill on the compact cache row. |
| 0008 | Mistral's DSpark draft head, split over the four Sparks. |
| 0009 | The check of drafts with the exact arithmetic of a normal step (replies with drafts = replies without). |
| 0010 | The attention block as one chain of exchanges between the Sparks, also for contexts above 8,192 tokens. |
| 0011 | Less prefill memory: the 16-bit matrices are unpacked one layer at a time. |
| 0012 | Server mode with the draft head, and prefill buffers that work with the shared cache. |
| 0013 | Buffers for steps that hold several requests. |
| 0014 | The API front end for Mistral Large 4: reasoning blocks and tool calls. |
| 0015 | Less prefill memory: one shared buffer in place of one for each layer (0.86 → 0.15 GB at 16K context). Reasoning tokens are counted in the usage. |
| 0016 | Requests longer than 8,192 tokens in server mode. Correct prompt-speed metrics. |
| 0017 | A prefill counter in the API metrics. |
| 0018 | Prefill attention in parts of 1,024 rows (smaller work memory). |
| 0019 | Prefill: each Spark scores one quarter of the attention index. |
| 0020 | The draft head reads 32 rows of prompt context in one step. |
| 0021 | Prefill sparse attention with shapes that keep more exchanges in flight. |
| 0022 | Decode: each Spark computes one quarter of the attention index query. |
| 0023 | The API can accept more than one model name. |
| 0024 | Decode attention for all heads of a Spark in one kernel. |
| 0025 | The draft head reports its probabilities; the engine selects the draft length from them. |
| 0026 | Weight records in the order that the kernels read them (`--wide`): a check step of 4 rows went from 90.6 to 83.1 ms. A prefill profile. |
| 0027 | More detail in the prefill profile. |
| 0028 | A build switch for an 8-bit attention cache row. Off by default. |
| 0029 | Two other designs of the prefill expert kernels and timing probes. Off by default. |

The engine is built with these flags (see `ENGINE_FLAGS` in `scripts/config.sh`):
`-DG6_NR=4 -DDEC_S2 -DG6_CP=1 -DG6_MODEL_ML4=1 -DML4_BENCH_NEXP=129 -DPF_FC2GL=6 -DPF_ISGRT=1 -DDEC_SLOT=2 -DDEC_AVP=1 -DSEL32 -DDEC_IDX31=1`

The engine runs with the settings in `engine_env` and `engine_args` in `scripts/serve.sh`.
