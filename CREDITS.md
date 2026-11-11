# Credits

This recipe connects the work of other people. It contains no model weights and no engine source; it downloads them.

- **Mistral AI** — [Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B), the model,
  and its DSpark draft head, which this recipe downloads from Mistral's repository.
- **Piotr Kosecki** — the
  [EXL3 3.0 bpw quant](https://huggingface.co/piotrkosecki/Mistral-Large-4-1T-A52B-EXL3-3.0bpw) of the model. This
  recipe uses these weights without changes. The mathematics of the draft head in patch 0008 follows the port of it in that repository.
- **turboderp** — [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the EXL3 format.
  `scripts/convert/exl3_ref.py` is a reference decoder written from the ExLlamaV3 sources.
- **Ash Hart and the TensorFold contributors** — [TensorFold](https://github.com/ashhart/TensorFold). The reference
  decoder also follows its EXL3 kernels.
- **The gr8 engine contributors** — [gr8](https://github.com/kindlingai/gr8), the inference engine. The patches in
  `patches/gr8` add Mistral Large 4 to it.
- **MiaAI-Lab** — the 4x DGX Spark recipes for GLM and DeepSeek. The ring inventory in `scripts/cluster.sh` and the
  memory counting in `scripts/memavail.sh` follow their scripts.
- **NVIDIA** — the DGX Spark, and the PyTorch container image in which the engine is built and runs.

The recipe and its patches: [@grubbychicken](https://x.com/grubbychicken).
