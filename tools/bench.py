#!/usr/bin/env python3
"""Benchmarks through the OpenAI API, timed on the client from the streamed reply. Python standard library only.
The same tool measures any OpenAI-compatible server, so you can compare servers with the same prompts.

Usage: bench.py [CELL ...]        Env: HOST (127.0.0.1), PORT (8000), MODEL (mistral-large-4)
Cells (default: all, in this order):
  decode      three prompts (edit code, write prose, write new code), 700 tokens each, temperature 0
  sampled     the same prompts at temperature 1, top_p 0.95, seed 7
  prefill     new prompts of about 2K, 4K, 8K, 16K, 28K, 48K and 62K tokens with four facts hidden in each;
              prefill speed = prompt tokens / time to the first token; the reply must contain the four facts
  longdecode  decode speed after a prompt of about 24K tokens
  turns       a three-turn chat about a 19K-token document: time to the first token of each turn
  beside      a short request sent 2 seconds after a 24K-token prompt
  conc        1, 2, 4 and 8 requests at the same time (different prompts), 300 tokens each
How the numbers are made:
  decode tok/s = (completion tokens - 1) / (time of the last chunk - time of the first chunk)
  ttft         = time from the request to the first chunk (reasoning or content)
  sha          = the first 12 hex digits of the SHA-256 of the reply text; equal values mean identical replies
  accepted     = draft tokens accepted / proposed, from the server's /metrics (only when one request runs)
"""
import hashlib, json, os, random, sys, threading, time, urllib.request

HOST, PORT = os.environ.get("HOST", "127.0.0.1"), os.environ.get("PORT", "8000")
MODEL = os.environ.get("MODEL", "mistral-large-4")
BASE = "http://%s:%s" % (HOST, PORT)
WORDS = "harbor copper meadow signal bracket tundra keystone ration pillar beacon clover digest fathom gravel hollow ingot juniper kettle lumen marble nectar onyx pylon quartz".split()
FACTS = [("The vault code is 7391.", "7391"), ("Maria owns a blue bicycle.", "blue"), ("The meeting moved to Thursday.", "Thursday"), ("The spare key is under the green flowerpot.", "green")]
CODE = '''"""inventory.py - a small stock ledger kept in a JSON file."""
import json
import os
import sys
import time


class Ledger:
    def __init__(self, path):
        self.path = path
        self.items = {}
        if os.path.exists(path):
            with open(path) as f:
                self.items = json.load(f)

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.items, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def add(self, name, count, price):
        if count <= 0:
            raise ValueError("count must be positive")
        if price < 0:
            raise ValueError("price must not be negative")
        item = self.items.setdefault(name, {"count": 0, "price": price, "log": []})
        item["count"] += count
        item["price"] = price
        item["log"].append([time.time(), "add", count])

    def remove(self, name, count):
        if name not in self.items:
            raise KeyError("no such item: " + name)
        item = self.items[name]
        if count > item["count"]:
            raise ValueError("not enough stock for " + name)
        item["count"] -= count
        item["log"].append([time.time(), "remove", count])

    def value(self):
        return sum(i["count"] * i["price"] for i in self.items.values())

    def report(self, out=sys.stdout):
        for name in sorted(self.items):
            item = self.items[name]
            out.write("%-20s %6d  %10.2f\\n" % (name, item["count"], item["count"] * item["price"]))
        out.write("%-20s %6s  %10.2f\\n" % ("total", "", self.value()))


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: inventory.py FILE add NAME COUNT PRICE | remove NAME COUNT | report\\n")
        return 2
    ledger = Ledger(argv[1])
    command = argv[2]
    if command == "add":
        ledger.add(argv[3], int(argv[4]), float(argv[5]))
    elif command == "remove":
        ledger.remove(argv[3], int(argv[4]))
    elif command == "report":
        ledger.report()
        return 0
    else:
        sys.stderr.write("unknown command: " + command + "\\n")
        return 2
    ledger.save()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''
PROMPTS = [
    ("code edit", "Here is a Python file:\n\n" + CODE + "\n\nRewrite the whole file with the docstring moved into a comment block at the top and every error message prefixed with ERROR:. Output only the full file."),
    ("prose", "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle."),
    ("code new", "Write a Python function that parses an ISO 8601 date string into a datetime without using any libraries, with tests."),
]


def metrics():
    out = {}
    try:
        for line in urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode().splitlines():
            for k in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
                if line.startswith("vllm:" + k):
                    out[k] = float(line.split()[-1])
    except Exception:
        pass
    return out


def chat(messages, max_tokens, sampled=False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    body.update({"temperature": 1.0, "top_p": 0.95, "seed": 7} if sampled else {"temperature": 0})
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, last, text, content, usage = time.time(), None, None, "", "", {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if piece:
                    last = time.time()
                    first = first or last
                    text += piece
                    content += d.get("content") or ""
    n = usage.get("completion_tokens", 0)
    return {"prompt": usage.get("prompt_tokens", 0), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "n": n, "ttft": (first or time.time()) - t0, "rate": (n - 1) / (last - first) if first and last > first else 0.0,
            "wall": time.time() - t0, "sha": hashlib.sha256(text.encode()).hexdigest()[:12], "content": content}


def show(name, r, m0=None, m1=None):
    acc = ""
    if m0 and m1:
        acc = "  accepted %d/%d" % (m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"],
                                    m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"])
    print("%-10s prompt %5d tokens (cached %s)  reply %4d tokens  ttft %6.2f s  decode %6.2f tok/s  total %6.1f s%s  sha %s"
          % (name, r["prompt"], r["cached"], r["n"], r["ttft"], r["rate"], r["wall"], acc, r["sha"]), flush=True)


def filler(words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS[:12]) for _ in range(words))


def cell_decode(sampled=False):
    for name, prompt in PROMPTS:
        m0 = metrics(); r = chat(prompt, 700, sampled); show(name, r, m0, metrics())


def cell_prefill():
    bad = 0
    for target in (1300, 2600, 5200, 10400, 18000, 30000, 39000):
        rng = random.Random(target)
        chunks = [" ".join(rng.choice(WORDS) for _ in range(target // len(FACTS))) for _ in FACTS]
        doc = " ".join(c + " " + f for c, (f, _) in zip(chunks, FACTS))
        q = "What is the vault code, what colour is Maria's bicycle, when is the meeting, and what colour is the flowerpot over the spare key? One line."
        r = chat(doc + "\n" + q, 400)
        hits = sum(a.lower() in r["content"].lower() for _, a in FACTS)
        bad += hits < 4
        print("prefill    prompt %5d tokens (cached %s)  ttft %6.2f s  prefill %6.1f tok/s  facts %d/4"
              % (r["prompt"], r["cached"], r["ttft"], r["prompt"] / r["ttft"] if r["ttft"] else 0, hits), flush=True)
    return bad


def cell_longdecode():
    m0 = metrics()
    r = chat(filler(18000, 5) + "\nIgnore the words above. Write a 300-word story about a lighthouse.", 700)
    show("long", r, m0, metrics())


def cell_turns():
    rng = random.Random(5)
    doc = " ".join(rng.choice(WORDS) for _ in range(9000)) + " The vault code is 7391. " + " ".join(rng.choice(WORDS) for _ in range(3000))
    msgs = [{"role": "user", "content": doc + "\nWhat is the vault code? One line."}]
    for i, q in enumerate([None, "And what colour is a clover, usually? One line.", "Thanks. Repeat the vault code once more."]):
        if q:
            msgs.append({"role": "user", "content": q})
        r = chat(msgs, 300)
        print("turn %d     prompt %5d tokens (cached %s)  ttft %6.2f s  | %s" % (i + 1, r["prompt"], r["cached"], r["ttft"], r["content"].strip()[:60]), flush=True)
        msgs.append({"role": "assistant", "content": r["content"]})


def cell_beside():
    res = {}
    big = threading.Thread(target=lambda: res.update(big=chat(filler(18000, 11) + "\nIgnore the words above. Reply with the single word: done.", 60)))
    big.start(); time.sleep(2)
    res["small"] = chat("Name three primary colours. One line.", 120)
    show("short", res["small"]); big.join(); show("long", res["big"])


def cell_conc():
    topics = ["a lighthouse", "a mountain railway", "a night market", "a desert well", "a glass factory", "a river ferry", "an old observatory", "a winter orchard"]
    for n in (1, 2, 4, 8):
        for rep in (1, 2):
            res = [None] * n

            def one(i):
                res[i] = chat("Write a 250-word story about %s. Run %d-%d." % (topics[i % len(topics)], n, rep), 300)
            t0 = time.time()
            th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [t.start() for t in th]; [t.join() for t in th]
            wall = time.time() - t0
            rates, ttfts, toks = sorted(r["rate"] for r in res), [r["ttft"] for r in res], sum(r["n"] for r in res)
            print("conc %d rep %d  total %6.2f tok/s  each stream %5.2f / %5.2f / %5.2f tok/s (min / median / max)  ttft %5.2f / %5.2f s (mean / max)"
                  % (n, rep, toks / wall, rates[0], rates[len(rates) // 2], rates[-1], sum(ttfts) / n, max(ttfts)), flush=True)


CELLS = {"decode": cell_decode, "sampled": lambda: cell_decode(True), "prefill": cell_prefill, "longdecode": cell_longdecode,
         "turns": cell_turns, "beside": cell_beside, "conc": cell_conc}


def main():
    cells = sys.argv[1:] or list(CELLS)
    if any(c not in CELLS for c in cells):
        sys.exit(__doc__)
    bad = 0
    for c in cells:
        print("== " + c, flush=True)
        bad += CELLS[c]() or 0
    sys.exit(1 if bad else 0)


main()
