# Changelog

## 1.0.0

First release.

- One command (`./start.sh`) finds the four Sparks, checks the ring, builds the engine, downloads and converts the
  weights, starts the server and tests it.
- gr8 at commit `069ec213` with 29 patches for Mistral Large 4 (`patches/gr8`).
- OpenAI-compatible API with reasoning, tools, streaming, 65,536 tokens of context and four requests at a time.
- `tools/bench.py` and `tools/sanity.py`.
