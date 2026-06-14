#!/usr/bin/env python3
"""
T-bench runner — open arms (raw, oracle, naive_rag) + bring-your-own-system.

Each probe is a standalone question answered ONLY from the context the chosen
arm builds from the conversation history. Answers are scored deterministically
(see score.py). Results are written to results/<run-id>_<arm>_<provider>.jsonl
and the run is resume-safe (already-scored probes are skipped).

Examples:
  # plumbing test, no API key (mock answerer emits gold-bearing answers):
  python run_tbench.py --data ../data/v1.1 --arm raw --provider mock --limit 5

  # real run against an OpenAI-compatible endpoint:
  python run_tbench.py --data ../data/v1.1 --arm naive_rag \\
      --provider openai --model gpt-5-mini

  # benchmark your own system:
  python run_tbench.py --data ../data/v1.1 --arm byo --byo my_system.py \\
      --provider openai --model gpt-5-mini

API keys are read from the environment (OPENAI_API_KEY / OPENROUTER_API_KEY).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score                                     # noqa: E402
import arms as armlib                                       # noqa: E402

SYSTEM_PROMPT = (
    "You are a personal assistant continuing a long-running relationship "
    "with the user. The conversation history below is authentic. Answer the "
    "final question only from that history. If the history does not contain "
    "the answer, say plainly that the user never mentioned it. Be concise."
)

_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",
                   "OPENROUTER_API_KEY"),
}


def build_arm(args):
    if args.arm == "raw":
        return armlib.arm_raw
    if args.arm == "oracle":
        return armlib.arm_oracle
    if args.arm == "naive_rag":
        return armlib.NaiveRAG(top_k=args.top_k)
    if args.arm == "byo":
        if not args.byo:
            raise SystemExit("--arm byo requires --byo path/to/your_file.py")
        return armlib.load_byo(args.byo)
    raise SystemExit(f"unknown arm {args.arm}")


def ask_mock(probe: dict, wrong: bool = False) -> str:
    """Plumbing-test answerer: emits a gold-bearing (or deliberately wrong)
    answer so you can verify the pipeline end-to-end without an API key."""
    if wrong:
        return "I am not sure, possibly 42 or Thursday."
    g = probe["gold"]
    if "contains_any" in g:
        return f"The answer is {g['contains_any'][0]}."
    if "gold_set" in g:
        return "We also discussed " + " and ".join(g["gold_set"]) + "."
    if "groups" in g:
        return ("Apparently it is the " + g["groups"][0][0] +
                " floor, but that's just what you heard — not confirmed.")
    if "abstain_markers" in g:
        return "You never mentioned that figure, so I have no record of it."
    if "source" in g:
        s = g["source"]
        if s == "user":
            return "You mentioned it yourself, you first said it early on."
        if s == "assistant":
            return "I worked that out from the figures you gave me."
        return f"You heard it from {s.split(':', 1)[1]}."
    return "(mock: no strategy)"


def ask_model(context: str, question: str, model: str, provider: str) -> tuple[str, dict]:
    url, env_key = _ENDPOINTS[provider]
    key = os.environ.get(env_key)
    if not key:
        raise SystemExit(f"{env_key} not set in the environment")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"=== CONVERSATION HISTORY ===\n{context}\n"
                        f"=== QUESTION ===\n{question}"},
        ],
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"], out.get("usage", {})
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as e:
            code = getattr(e, "code", None)
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
            transient = code is None or code >= 500 or code == 429
            if attempt < 4 and transient:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"API {code}: {detail}") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir (conversations.json + probes.json)")
    ap.add_argument("--arm", required=True,
                    choices=["raw", "oracle", "naive_rag", "byo"])
    ap.add_argument("--byo", help="path to your build_context file (--arm byo)")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "mock-wrong", "openai", "openrouter"])
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--top-k", type=int, default=8, help="naive_rag turns kept")
    ap.add_argument("--limit", type=int, default=0, help="first N probes only")
    ap.add_argument("--run-id", default="run", help="output file prefix")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = Path(__file__).parent / args.data
    convs = {c["conv_id"]: c for c in
             json.loads((data / "conversations.json").read_text())}
    probes = json.loads((data / "probes.json").read_text())
    if args.limit:
        probes = probes[:args.limit]

    arm = build_arm(args)

    res_dir = Path(__file__).resolve().parent.parent / "results"
    res_dir.mkdir(exist_ok=True)
    out_path = res_dir / f"{args.run_id}_{args.arm}_{args.provider}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                (done.discard if r.get("error") else done.add)(r["probe_id"])

    todo = [p for p in probes if p["probe_id"] not in done]
    n, total = 0, 0.0
    t0 = time.time()
    with out_path.open("a") as fh:
        for p in probes:
            if p["probe_id"] in done:
                continue
            ctx = arm(convs[p["conv_id"]], p)
            err = None
            try:
                if args.provider.startswith("mock"):
                    ans, usage = ask_mock(p, args.provider == "mock-wrong"), {}
                else:
                    ans, usage = ask_model(ctx, p["question"], args.model, args.provider)
                    time.sleep(0.3)
            except (RuntimeError, KeyError, ValueError) as e:
                err = str(e)[:200]
                ans, usage = "", {}
                print(f"  x {p['probe_id']} skipped: {err}", flush=True)
            s = score(p, ans) if not err else 0.0
            n += 1
            total += s
            fh.write(json.dumps({
                "probe_id": p["probe_id"], "family": p["family"],
                "subtype": p["subtype"], "arm": args.arm,
                "provider": args.provider, "model": args.model,
                "score": s, "answer": ans, "usage": usage,
                "ctx_tokens": len(ctx) // 4, "error": err,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{n}/{len(todo)}] mean={total / n:.3f} elapsed={el:.1f}m",
                      flush=True)

    if n:
        print(f"{args.arm}/{args.provider}: {n} probes, mean score {total / n:.3f}")
        print(f"  -> {out_path.name}")
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        fams: dict[str, list] = {}
        for r in rows:
            fams.setdefault(r["family"], []).append(r["score"])
        for f in sorted(fams):
            v = fams[f]
            print(f"  {f}: {sum(v) / len(v):.3f}  (n={len(v)})")
    else:
        print("nothing to do (all probes already scored)")


if __name__ == "__main__":
    main()
