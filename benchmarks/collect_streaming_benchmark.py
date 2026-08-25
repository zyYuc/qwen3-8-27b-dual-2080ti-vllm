#!/usr/bin/env python3
"""Streaming first-character benchmark.

TTFT ends at the first non-empty reasoning_content, reasoning, or content
delta. For Qwen thinking mode, the first streamed thinking character counts
as first character; this does not wait for the final answer body.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("system performance optimization architecture memory bandwidth latency "
         "throughput parallel computation kernel buffer cache scheduling allocation "
         "fragmentation synchronization inference quantization compression precision "
         "stability reliability scalability bottleneck utilization").split()


def prompt(target_tokens, seed):
    rng = random.Random(seed)
    tokens = [rng.choice(WORDS) for _ in range(target_tokens)]
    return "Unique benchmark context id: {}. Summarize this context.\n\n{}".format(
        seed, " ".join(tokens))


def request_once(base_url, model, text, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "stream": True, "temperature": 0.6,
            "top_p": 0.95, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    first = None
    usage = {}
    buffer = b""
    with urllib.request.urlopen(req, timeout=900) as response:
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                delta = (data.get("choices") or [{}])[0].get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    first = time.perf_counter()
                if data.get("usage"):
                    usage = data["usage"]
    ended = time.perf_counter()
    ttft = first - started if first else None
    decode_time = ended - first if first else None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return {
        "status": 200, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(ended - started, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
        "decode_tok_s": round(completion_tokens / decode_time, 1)
        if completion_tokens and decode_time else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[4000, 8000, 20000])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="benchmark-result.json")
    args = parser.parse_args()
    results = []
    for target in args.prompt_tokens:
        for run in range(1, args.runs + 1):
            result = request_once(args.base_url, args.model, prompt(target, target * 100 + run), args.max_tokens)
            result["target_prompt_tokens"] = target
            result["run"] = run
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(results, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
