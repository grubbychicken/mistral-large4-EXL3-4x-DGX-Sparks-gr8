# Notice

Mistral Large 4 on four DGX Sparks with the gr8 engine.
Copyright 2026 the authors of this repository.

- The scripts, tools and documents of this repository have the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
- The files in `patches/gr8` are changes to the gr8 engine and contain parts of it. They have gr8's licence (MIT):
  see [patches/gr8/LICENSE](patches/gr8/LICENSE), copyright (c) 2026 gr8 engine contributors.
- Some files contain work of other projects. See "Work of other projects in this repository" below.

This repository contains **no model weights, no engine source and no binaries**. `start.sh` downloads them. Each
has its own terms:

| What | Terms | Note |
|---|---|---|
| Mistral-Large-4-1T-A52B (the weights and the draft head) | Mistral Ecosystem Open-Weight License (the `LICENSE` file in the download) | Read it before you use the model. If you give other people access to the model as a commercial service, you need a separate licence from Mistral. |
| The EXL3 3.0 bpw quant | The terms in its Hugging Face repository | A quantized copy of Mistral's weights; Mistral's licence applies too. |
| gr8 | MIT | Downloaded from GitHub at a fixed commit. |
| `nvcr.io/nvidia/pytorch` container image | NVIDIA Deep Learning Container License | The terms are inside the image. |
| The Rust container image | The terms of the Docker official image and of Rust | Used only for the build. |
| Hugging Face `transformers` | Apache-2.0 | Installed in the converter's container. |

If you give this recipe or a changed version to other people, keep this notice and say what you changed.

This project is independent. Mistral AI, NVIDIA, KindlingAI (gr8) and the other persons and projects named here did
not make it and do not endorse it. Their names and marks belong to them.

## Work of other projects in this repository

**MiaAI-Lab DGX Spark recipes (Apache License 2.0).** `scripts/memavail.sh` and the functions `node_inventory` and
`pair_links` in `scripts/cluster.sh` come from `scripts/memavail.sh`, `scripts/tp4.sh` and `scripts/nodes.sh` of the
MiaAI-Lab recipes (https://github.com/MiaAI-Lab). They were changed for this recipe (other names, messages and
checks). Their notice: "Copyright 2026 Mia's AI Lab (MiaAI-Lab). Licensed under the Apache License, Version 2.0."
The licence text is the same as [LICENSE](LICENSE).

**ExLlamaV3 and TensorFold (MIT).** `scripts/convert/exl3_ref.py` is a decoder for the EXL3 format in Python. It
was written from the sources of ExLlamaV3 (https://github.com/turboderp-org/exllamav3) and of the EXL3 kernels of
TensorFold (https://github.com/ashhart/TensorFold, code with TensorFold's MIT notice). Their notices:

    Copyright (c) 2025 Turboderp
    Copyright (c) 2026 TensorFold contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

