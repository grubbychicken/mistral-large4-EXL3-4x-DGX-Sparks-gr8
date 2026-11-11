# Benchmarks

`tools/bench.py` measures the server through its API. It times everything on the client from the streamed reply,
so the values include the API and the network path on Spark 0. It uses only the Python standard library.

```bash
python3 tools/bench.py                 # all cells
python3 tools/bench.py decode conc     # only the cells that you name
PORT=8000 MODEL=mistral-large-4 python3 tools/bench.py decode
```

## Cells

| Cell | What it sends | What it reports |
|---|---|---|
| `decode` | Three prompts (edit a Python file, write a short story, write a new function). 700 tokens each, temperature 0. | Decode speed, time to the first token, accepted drafts, a hash of the reply. |
| `sampled` | The same prompts at temperature 1, top_p 0.95, seed 7. | The same values. |
| `prefill` | New prompts of about 2K to 62K tokens. Four facts are hidden in each prompt. | Prompt speed, and how many of the four facts the reply contains. |
| `longdecode` | A prompt of about 24K tokens, then a 300-word story. | Decode speed deep in the context. |
| `turns` | A three-turn chat about a 19K-token document. | Time to the first token of each turn, and the number of reused tokens. |
| `beside` | A short request 2 seconds after a 24K-token prompt. | How long the short request waits. |
| `conc` | 1, 2, 4 and 8 requests at the same time, 300 tokens each. | Total speed, speed of each request, time to the first token. |

## Definitions

- **Decode speed** = (completion tokens − 1) / (time of the last chunk − time of the first chunk).
- **Prompt speed (prefill)** = prompt tokens / time to the first token. The time includes one decode step, so the
  real prompt speed is a little higher.
- **Hash (`sha`)**: the first 12 hex digits of the SHA-256 of the reply text. Equal hashes mean identical replies.
  At temperature 0 the hash must be the same on each run and each start.
- **Accepted drafts**: draft tokens that the model accepted / draft tokens that the draft head proposed.

## How the README values were measured

- Four DGX Sparks in a ring, the default settings of this recipe, one benchmark client on Spark 0.
- Three fresh starts. The decode cells ran on each start; the other cells ran once.
- The memory and the GPU power of each Spark were sampled every 5 seconds (`/proc/meminfo`, `nvidia-smi`).
- The agreement with the full-precision model was measured outside this repository: teacher-forced log-probabilities
  of 64 sequences × 2,048 tokens (prose, code, chat, mathematics, other languages), compared with Mistral's FP8
  release from position 1,024 of each sequence.

## Rules for honest numbers

- Use new text for prefill measurements. The server reuses the start of a prompt that it saw before; the
  `cached` column shows how many tokens were reused.
- Run each measurement more than once. Compare the hashes at temperature 0.
- Stop other programs on the Sparks first.
