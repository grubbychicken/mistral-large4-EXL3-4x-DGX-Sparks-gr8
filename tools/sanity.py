#!/usr/bin/env python3
"""Four live checks against the served OpenAI-compatible API (this recipe's PORT): chat
greedy, streaming, a tool-call round trip, and reasoning_effort low. PASS/FAIL per check,
one attempt each -- nothing retries forever, so what you see is what the server did.

Usage: sanity.py HOST PORT [MODEL_NAME]
  HOST PORT   where the API listens, e.g. 127.0.0.1 8000 (PORT in scripts/config.sh; by default
              the API listens on 127.0.0.1 of the first Spark)
  MODEL_NAME  optional; when omitted it is taken from GET /v1/models (first id) and shown

The checks, and what PASS means (mechanism, not brilliance):
  chat greedy       POST /v1/chat/completions, temperature 0 / top_p 1.0 (greedy: these
                    are byte-stability gates; the model card's sampled recommendation,
                    temperature 1.0 / top_p 0.95, is a different mode). PASS = HTTP 200, a
                    well-formed choice with non-empty content and a finish_reason.
  streaming         the same shape with stream:true. PASS = at least one content chunk, a
                    finish_reason, and a `data: [DONE]` terminator. Whether the streamed
                    text equals the greedy reply is reported, not required (both are greedy
                    here, so a difference is worth a look).
  tool-call round   a tools=[...] request the model should answer by calling the tool;
                    PASS = a tool_calls message whose function name and JSON arguments
                    parse, then the tool result sent back and a final answer produced.
                    Whether the final answer uses the result is printed for you to judge.
  reasoning_effort  a request carrying reasoning_effort "low" (this recipe's
                    REASONING_EFFORT default; the API parses Mistral's reasoning blocks).
                    PASS = HTTP 200 and a well-formed completion. Any reasoning field in
                    the reply is reported.

Exit codes: 0 all checks PASS | 1 at least one FAIL | 2 server unreachable | 64 bad usage.
No output is a benchmark: these are ~50-token prompts (tools/bench.py measures speed). Stdlib
only (urllib). Python 3.9+.
"""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request

TIMEOUT_DEFAULT = 600        # generous: a first request can wait behind other work
GREEDY = {"temperature": 0, "top_p": 1.0}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "count_words",
        "description": "Count the words in a short phrase.",
        "parameters": {
            "type": "object",
            "properties": {"phrase": {"type": "string",
                                      "description": "The phrase whose words are counted."}},
            "required": ["phrase"],
        },
    },
}]


class ArgsParser(argparse.ArgumentParser):
    # argparse exits 2 on usage errors by default; 2 here means "server unreachable",
    # so bad flags leave through 64. Exit codes are in the docstring.
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(64, "%s: error: %s\n" % (self.prog, message))


def base_url(host, port):
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host          # bare IPv6 address
    return "http://%s:%s/v1" % (host, port)


def http_error_body(exc):
    try:
        return exc.read().decode("utf-8", "replace")[:300].strip()
    except Exception:
        return ""


def post_json(url, payload, timeout):
    """One POST, one attempt. Returns the parsed JSON. Raises on connection and HTTP errors."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def stream_chat(url, payload, timeout):
    """One streaming POST, one attempt. Returns a summary dict (Mia's harness shape):
    first-chunk latency, the concatenated text, finish_reason, the [DONE] sentinel."""
    body = dict(payload)
    body["stream"] = True
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter()
    ttft = None
    pieces = []
    finish = None
    saw_done = False
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                saw_done = True
                continue
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    pieces.append(piece)
                    chunks += 1
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    return {"ttft": ttft, "text": "".join(pieces), "finish": finish,
            "saw_done": saw_done, "chunks": chunks, "elapsed": time.perf_counter() - started}


def resolve_model(base, timeout):
    """GET /v1/models, first id. POST would 405 on most servers; a GET here, once."""
    req = urllib.request.Request(base + "/models")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    ids = [m.get("id") for m in data.get("data") or [] if m.get("id")]
    return ids[0] if ids else None


class Result:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.detail = ""


def check_chat_greedy(base, model, timeout):
    """[1/4] -- greedy chat, the shape every other check builds on."""
    res = Result("chat greedy")
    payload = {"model": model, "messages": [{"role": "user",
                                             "content": "Respond with the single word: pong"}],
               "max_tokens": 512, **GREEDY}
    t0 = time.perf_counter()
    data = post_json(base + "/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - t0
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    finish = choice.get("finish_reason")
    if not content or not str(content).strip():
        res.detail = "HTTP 200 but empty message.content (finish_reason=%r)" % (finish,)
        return res, None
    res.passed = finish is not None
    res.detail = "%d chars in %.2f s, finish_reason=%s%s" % (
        len(str(content)), elapsed, finish,
        "  reply: %s" % str(content)[:80].replace("\n", " "))
    return res, str(content)


def check_streaming(base, model, timeout, greedy_text):
    """[2/4] -- SSE streaming: content chunks, finish_reason, [DONE]."""
    res = Result("streaming")
    payload = {"model": model, "messages": [{"role": "user",
                                             "content": "Respond with the single word: pong"}],
               "max_tokens": 512, **GREEDY}
    t0 = time.perf_counter()
    out = stream_chat(base + "/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - t0
    if not out["text"]:
        res.detail = "stream ended with no content chunks (finish_reason=%r, [DONE]=%s)" % (
            out["finish"], out["saw_done"])
        return res, None
    problems = []
    if not out["saw_done"]:
        problems.append("no [DONE] terminator")
    if not out["finish"]:
        problems.append("no finish_reason in any chunk")
    if out["ttft"] is None:
        problems.append("no timed content chunk")
    matches = greedy_text is not None and out["text"].strip() == greedy_text.strip()
    res.passed = not problems
    res.detail = "%d chunks in %.2f s, first chunk after %.2f s%s%s  streamed: %s" % (
        out["chunks"], elapsed, out["ttft"] or -1.0,
        "; " + "; ".join(problems) if problems else "",
        ", byte-identical to the greedy reply" if matches else "",
        out["text"][:60].replace("\n", " "))
    return res, out["text"]


def check_tool_roundtrip(base, model, timeout):
    """[3/4] -- the model calls the tool, the result goes back, a final answer comes out."""
    res = Result("tool-call round trip")
    phrase = "the quick brown fox jumps"
    messages = [{"role": "user", "content":
                 "Use the count_words tool on the phrase '%s', then tell me the count "
                 "in one short sentence." % phrase}]
    payload = {"model": model, "messages": messages, "tools": TOOLS,
               "tool_choice": "auto", "max_tokens": 128, **GREEDY}
    first = post_json(base + "/chat/completions", payload, timeout)
    choice = (first.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        res.detail = "no tool_calls in the reply (content=%r)" % (str(message.get("content"))[:80],)
        return res, None
    call = calls[0]
    fn = call.get("function") or {}
    name = fn.get("name")
    raw_args = fn.get("arguments")
    try:
        args_obj = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
    except ValueError:
        res.detail = "tool called (%s) but arguments are not JSON: %r" % (name, raw_args)
        return res, None
    if name != "count_words":
        res.detail = "called %r instead of count_words (arguments %r)" % (name, args_obj)
        return res, None

    # The round trip: the assistant's tool_calls message and the tool result go back in.
    messages.append(message)
    messages.append({"role": "tool",
                     "tool_call_id": call.get("id") or "call_0",
                     "content": "%d" % len(phrase.split())})
    second = post_json(base + "/chat/completions",
                       {"model": model, "messages": messages, "tools": TOOLS,
                        "max_tokens": 128, **GREEDY}, timeout)
    final = ((second.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not final or not str(final).strip():
        res.detail = "round trip completed (%s %r) but no final answer came back" % (name, args_obj)
        return res, None
    uses_result = "5" in str(final)          # the phrase's true word count
    res.passed = True
    res.detail = "called %s(%r) -> tool result sent -> final answer%s: %s" % (
        name, args_obj,
        " (mentions 5)" if uses_result else " (does not mention 5 -- judge from the text)",
        str(final)[:80].replace("\n", " "))
    return res, None


def check_reasoning_effort(base, model, timeout):
    """[4/4] -- reasoning_effort low is accepted and answered (the port's reasoning parser)."""
    res = Result("reasoning_effort low")
    payload = {"model": model, "messages": [{"role": "user",
                                             "content": "Reply with one short sentence."}],
               "max_tokens": 64, "reasoning_effort": "low", **GREEDY}
    data = post_json(base + "/chat/completions", payload, timeout)
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not content or not str(content).strip():
        res.detail = "HTTP 200 but empty message.content"
        return res, None
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    res.passed = choice.get("finish_reason") is not None
    res.detail = "accepted (HTTP 200, finish_reason=%s)%s" % (
        choice.get("finish_reason"),
        "; reasoning field in reply: %d chars" % len(str(reasoning)) if reasoning else
        "; no reasoning field in the reply")
    return res, None


def main(argv=None):
    parser = ArgsParser(
        description="Four live checks against the served API, PASS/FAIL each, one attempt "
                    "each (see the module docstring).",
        epilog="Greedy and small on purpose: these are gates, not benchmarks. start.sh "
               "runs this after the health wait.")
    parser.add_argument("host", help="API host, e.g. 127.0.0.1 (scripts/config.sh HOST)")
    parser.add_argument("port", type=int, help="API port (scripts/config.sh PORT, default 8000)")
    parser.add_argument("model_name", nargs="?", default=None,
                        help="the served model name (default: first id from /v1/models)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_DEFAULT,
                        help="seconds per request (default %d)" % TIMEOUT_DEFAULT)
    args = parser.parse_args(argv)
    if not (0 < args.port < 65536):
        parser.error("port must be 1..65535")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")

    base = base_url(args.host, args.port)
    # Reachability and the model name first: if the server is not there, say so and stop
    # (exit 2) instead of printing four failures against a dead port.
    try:
        model = args.model_name
        if model is None:
            model = resolve_model(base, min(args.timeout, 60))
        if not model:
            print("error: /v1/models returned no model ids; pass MODEL_NAME explicitly",
                  file=sys.stderr)
            return 2
        print("sanity: %s   model: %s" % (base, model))
    except urllib.error.HTTPError as exc:
        print("error: server answered HTTP %s: %s" % (exc.code, http_error_body(exc)),
              file=sys.stderr)
        return 2
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
        print("error: cannot reach %s: %s" % (base, exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print("error: server reply is not JSON: %s" % exc, file=sys.stderr)
        return 2

    # One attempt per check, in order; a connection that dies mid-run becomes a FAIL for
    # the check in flight rather than a crash, so the summary still shows what worked.
    started = time.perf_counter()
    try:
        greedy_reply = None
        results = []
        r1, greedy_reply = check_chat_greedy(base, model, args.timeout)
        results.append(r1)
        r2, _ = check_streaming(base, model, args.timeout, greedy_reply)
        results.append(r2)
        results.append(check_tool_roundtrip(base, model, args.timeout)[0])
        results.append(check_reasoning_effort(base, model, args.timeout)[0])
    except urllib.error.HTTPError as exc:
        results.append(_failed_result("late request", "HTTP %s: %s"
                                      % (exc.code, http_error_body(exc))))
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
        results.append(_failed_result("late request", "connection error: %s" % exc))
    except ValueError as exc:
        results.append(_failed_result("late request", "reply is not JSON: %s" % exc))

    print()
    for i, res in enumerate(results, 1):
        print("[%d/4] %-22s %s   %s" % (i, res.name, "PASS" if res.passed else "FAIL",
                                        res.detail))
    passed = sum(1 for r in results if r.passed)
    print()
    print("summary: %d/4 PASS in %.1f s" % (passed, time.perf_counter() - started))
    if passed == 4:
        print("the four mechanisms answer; this says nothing about speed -- that is "
              "tools/bench.py's job")
    return 0 if passed == 4 else 1


def _failed_result(name, detail):
    res = Result(name)
    res.detail = detail
    return res


if __name__ == "__main__":
    sys.exit(main())
