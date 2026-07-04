#!/usr/bin/env python3
"""
T-bench v1 runner — v0 (arms: t0_raw, t1_oracle; providers: mock, openai).

Engine arms (t2_tulbase, t3_tul20, t4_tul21) plug in via build_context()
adapters in a later version — bench-local engine lives in
comp-proof_v1/tulngin (retrieval.py) and gets copied here, not imported.

Per EXPERIMENT-PROTOCOL.md: every real run needs a RUNS.md row first.
Checkpointed: results/<run_id>_<arm>.jsonl, resume-safe (skips done probes).

Usage:
  python run_tbench.py --data ../data/pilot --arm t0_raw --provider mock
  python run_tbench.py --data ../data/pilot --arm t0_raw --provider openai \
      --model gpt-5-mini --limit 3            # smoke
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score                                     # noqa: E402

ENV_PATH = Path(__file__).resolve().parents[4] / ".env"     # comp-work/.env

SYSTEM_PROMPT = (
    "You are a personal assistant continuing a long-running relationship "
    "with the user. The conversation history below is authentic. Answer the "
    "final question only from that history. If the history does not contain "
    "the answer, say plainly that the user never mentioned it. Be concise."
)

# v0.6 controls (F6 no_history twins): NEUTRAL on purpose. The abstain-style
# prompt above would artificially crater the control arm and inflate C6_gain;
# here the model is INVITED to guess so the twin honestly measures whether
# probes are answerable on general grounds (DESIGN-v2 sections 3 and 9).
NO_HISTORY_PROMPT = (
    "You are a helpful personal assistant. No prior conversation history is "
    "available for this request. Answer the question as best you can on "
    "general grounds; if a definite answer would normally require personal "
    "context, make your best reasonable guess rather than refusing. "
    "Be concise."
)

# Product-flow fetch protocol (compressed arms only): mirrors the proxy's
# fetch_compressed tool with a plain-text handshake.
FETCH_PROTOCOL = (
    " The history may contain elided blocks marked like 'ID=compr-...'. "
    "If the answer requires an elided block, reply with exactly "
    "'FETCH:<id>' (nothing else) and you will receive the block content."
)
FETCH_RE = re.compile(r"^\s*FETCH:\s*([\w\-\.]+)", re.IGNORECASE)


# ----------------------------------------------------------------- contexts

def render_turns(sessions, keep=None):
    """Flatten sessions to 'turn lines'; keep=None -> all, else set of turn#s
    (session opener lines of kept turns are always included for day anchors
    ONLY when their turn is in keep — oracle fairness is opener-inclusive
    because generator lists openers in evidence when they matter)."""
    lines = []
    for s in sessions:
        for t in s["turns"]:
            if keep is None or t["turn"] in keep:
                lines.append(f"[turn {t['turn']} | {t['role']}] {t['text']}")
    return "\n".join(lines)


def build_context(conv: dict, probe: dict, arm: str, engine=None) -> str:
    if arm == "t0_raw":
        return render_turns(conv["sessions"])
    if arm == "t1_oracle":
        return render_turns(conv["sessions"], keep=set(probe["evidence_turns"]))
    if arm == "t2_tulbase":
        return engine.t2_tulbase(conv, probe)
    if arm == "t3_tul20":
        return engine.t3_tul20(conv, probe)
    if arm == "t4_tul21":
        return engine.t4_tul21(conv, probe)
    if arm == "t5_prod":
        return engine.t5_prod(conv, probe)
    if arm == "t6_tul21p0":
        return engine.t6_tul21p0(conv, probe)
    raise SystemExit(f"unknown arm {arm}")


# ----------------------------------------------------------------- providers

def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ask_mock(context: str, question: str, probe: dict, wrong: bool) -> str:
    """Plumbing-test provider: emits a gold-bearing (or wrong) answer.
    Fetch-aware: on anchor probes against a compressed context it first
    emits FETCH:<id> (exercising the fetch round), then answers gold."""
    if wrong:
        return "I am not sure, possibly 42 or Thursday."
    if (probe["subtype"].startswith("anchor")
            and "ID=compr-" in context
            and "FETCHED BLOCK" not in context):
        m = re.search(r"ID=([\w\-\.]+)", context)
        if m:
            return f"FETCH:{m.group(1).rstrip('.')}"
    g = probe["gold"]
    if "contains_any" in g:
        return f"The answer is {g['contains_any'][0]}."
    if "gold_set" in g:
        return "We also discussed " + " and ".join(g["gold_set"]) + "."
    if "groups" in g:
        # generic: one item from EVERY group (v0.6 golds are multi-group
        # beyond the original F5 hearsay shape); hedge tail keeps the old
        # F5 calibration behaviour harmless elsewhere
        return ("Putting it together: "
                + "; ".join(gr[0] for gr in g["groups"])
                + " — apparently so, though not officially confirmed.")
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


_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",
                   "OPENROUTER_API_KEY"),
}


def ask_openai(context: str, question: str, model: str,
               extra_system: str = "", provider: str = "openai",
               system_prompt: str | None = None) -> tuple[str, dict]:
    url, env_key = _ENDPOINTS[provider]
    key = os.environ.get(env_key)
    if not key:
        raise SystemExit(f"{env_key} not set (comp-work/.env)")
    user_content = (f"=== CONVERSATION HISTORY ===\n{context}\n"
                    f"=== QUESTION ===\n{question}") if context else question
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system",
             "content": (system_prompt if system_prompt is not None
                         else SYSTEM_PROMPT + extra_system)},
            {"role": "user", "content": user_content},
        ],
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    # Geçici hatalarda (5xx/522/429/timeout) backoff'lu retry —
    # tbench-007'de tek 522, T4 koşusunu düşürmüştü.
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            usage = out.get("usage", {})
            return out["choices"][0]["message"]["content"], usage
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as e:
            code = getattr(e, "code", None)
            # 4xx (429 hariç) kalıcı → hata gövdesini oku, çağırana ilet
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
            transient = code is None or code >= 500 or code == 429
            if attempt < 4 and transient:
                wait = 2 ** attempt * 3            # 3,6,12,24 sn
                print(f"  ! API hatası ({code or type(e).__name__}), "
                      f"{wait}sn sonra tekrar ({attempt + 1}/4)", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"API {code}: {detail}") from e


# ---------------------------------------------------------------------- main

def run_controls(args):
    """v0.6: run F6 no_history control twins (controls.json) exactly once —
    arm-independent by design (no history -> nothing to compress/retrieve).
    C6_gain = acc(grounded, per arm) - acc(no_history)  [DESIGN-v2 s3/s9]."""
    data = Path(__file__).parent / args.data if not Path(args.data).is_absolute() \
        else Path(args.data)
    ctrls = json.loads((data / "controls.json").read_text())
    if args.limit:
        ctrls = ctrls[:args.limit]
    if args.provider in ("openai", "openrouter"):
        load_env(ENV_PATH)
    model = (args.openrouter_model if args.provider == "openrouter"
             else args.model)
    res_dir = Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    out_path = res_dir / f"{args.run_id}_no_history_{args.provider}.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                (done.discard if r.get("error") else done.add)(r["probe_id"])
    n, total = 0, 0.0
    with out_path.open("a") as fh:
        for ct in ctrls:
            if ct["probe_id"] in done:
                continue
            err, usage = None, {}
            try:
                if args.provider.startswith("mock"):
                    ans = ask_mock("", ct["question"], ct,
                                   args.provider == "mock-wrong")
                else:
                    ans, usage = ask_openai(
                        "", ct["question"], model, provider=args.provider,
                        system_prompt=NO_HISTORY_PROMPT)
                    time.sleep(0.3)
            except (RuntimeError, KeyError, ValueError) as e:
                err, ans = str(e)[:200], ""
                print(f"  ✗ {ct['probe_id']} atlandı: {err}", flush=True)
            s = score(ct, ans) if not err else 0.0
            n += 1
            total += s
            fh.write(json.dumps({
                "probe_id": ct["probe_id"], "twin_of": ct["twin_of"],
                "family": ct["family"], "subtype": ct["subtype"],
                "arm": "no_history", "provider": args.provider,
                "model": model, "score": s, "answer": ans, "usage": usage,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"no_history controls: {n} run, mean {total / n:.3f}" if n
          else "controls: nothing to do")
    print(f"  -> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", required=False, default=None,
                    choices=["t0_raw", "t1_oracle", "t2_tulbase",
                             "t3_tul20", "t4_tul21", "t5_prod",
                             "t6_tul21p0"])
    ap.add_argument("--controls", action="store_true",
                    help="v0.6: run no_history control twins (controls.json) "
                         "instead of probes; arm-independent, run ONCE")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "mock-wrong", "openai", "openrouter"])
    ap.add_argument("--openrouter-model",
                    default="qwen/qwen-2.5-7b-instruct",
                    help="provider=openrouter ise cevaplayan model")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--limit", type=int, default=0, help="first N probes only")
    ap.add_argument("--families", default="",
                    help="virgüllü aile filtresi, örn. F6,F7 (015 amendment: "
                         "t2 yalnız yeni ailelerde — maliyet)")
    ap.add_argument("--run-id", default="dryrun",
                    help="RUNS.md id; 'dryrun' for plumbing tests")
    # t3 retrieval params — defaults = product policy (LongMemEval published)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--rel-frac", type=float, default=0.0)
    ap.add_argument("--min-k", type=int, default=8)
    args = ap.parse_args()

    if args.controls:
        return run_controls(args)
    if not args.arm:
        ap.error("--arm is required (unless --controls)")

    engine = None
    summ_mode = embed_mode = None
    real = args.provider in ("openai", "openrouter")
    if args.arm in ("t2_tulbase", "t3_tul20", "t4_tul21", "t5_prod",
                    "t6_tul21p0"):
        from engine_arms import (EngineArms, summarizer_mode,   # Mac-only
                                 embedder_mode)
        engine = EngineArms(threshold=args.threshold,
                            rel_frac=args.rel_frac, min_k=args.min_k)
        if args.arm in ("t2_tulbase", "t3_tul20", "t4_tul21"):
            summ_mode = summarizer_mode()
            print(f"summarizer: {summ_mode}")
            if args.arm == "t2_tulbase" and summ_mode != "lexrank" and real:
                raise SystemExit(
                    "T2 reddedildi: NLTK punkt yok, LexRank→first-N fallback. "
                    "Önce: python3 -c \"import nltk; "
                    "nltk.download('punkt_tab'); nltk.download('punkt')\"")
        if args.arm in ("t3_tul20", "t4_tul21", "t6_tul21p0"):
            embed_mode = embedder_mode()
            print(f"embedder: {embed_mode}")
            if embed_mode != "minilm" and real:
                raise SystemExit(
                    "T3/T4/T6 reddedildi: MiniLM yok → keep-all fallback. "
                    "comp-proof venv'ini aktive et.")
        if args.arm == "t5_prod":
            ready = engine.prod_retrieval_ready()
            print(f"t5_prod: üretim optimize_messages | _HAS_RETRIEVAL={ready}")
            if not ready and real:
                raise SystemExit(
                    "t5_prod reddedildi: proxy _HAS_RETRIEVAL=False → embedder "
                    "yok, retrieval min_k floor'a düşer (rel_frac anlamsız). "
                    "comp-proof venv'inde (sentence-transformers) koş.")

    data = Path(__file__).parent / args.data if not Path(args.data).is_absolute() \
        else Path(args.data)
    convs = {c["conv_id"]: c for c in
             json.loads((data / "conversations.json").read_text())}
    probes = json.loads((data / "probes.json").read_text())
    if args.families:
        fams = {f.strip() for f in args.families.split(",")}
        probes = [p for p in probes if p["family"] in fams]
    if args.limit:
        probes = probes[:args.limit]

    if args.provider in ("openai", "openrouter"):
        load_env(ENV_PATH)

    res_dir = Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    out_path = res_dir / f"{args.run_id}_{args.arm}_{args.provider}.jsonl"
    done = set()
    if out_path.exists():
        # error'lı satırları done SAYMA → resume'da yeniden denenir
        # (kalıcı 400 değilse düzelir; rapor probe_id bazında son-kayıt alır)
        for l in out_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("error"):
                done.discard(r["probe_id"])
            else:
                done.add(r["probe_id"])

    todo = [p for p in probes if p["probe_id"] not in done]
    n, total_score, in_chars = 0, 0.0, 0
    t_start = time.time()
    with out_path.open("a") as fh:
        for p in probes:
            if p["probe_id"] in done:
                continue
            ctx = build_context(convs[p["conv_id"]], p, args.arm, engine)
            in_chars += len(ctx)
            # v0.6.2: kanıt-kapsama kaydı — kayıp ayrıştırması için
            # (taşıma kaybı vs LLM kaybı; Abdullah 3 Tem). Render'lar turn
            # numarasını [T{i}] (i=global-1) ya da "[turn {n}" olarak taşır.
            ev = p.get("evidence_turns") or []
            if ev:
                got = {int(m) + 1 for m in re.findall(r"\[T(\d+)\]", ctx)}
                got |= {int(m) for m in re.findall(r"\[turn (\d+)", ctx)}
                ev_cov = round(sum(1 for t in ev if t in got) / len(ev), 3)
            else:
                ev_cov = None                      # abstain: kanıtsız probe
            sysprompt_extra = FETCH_PROTOCOL if args.arm == "t2_tulbase" else ""
            fetch_id, fetch_ok = None, None

            model = (args.openrouter_model if args.provider == "openrouter"
                     else args.model)

            def _ask(c):
                if args.provider.startswith("mock"):
                    return ask_mock(c, p["question"], p,
                                    args.provider == "mock-wrong"), {}
                a, u = ask_openai(c, p["question"], model,
                                  extra_system=sysprompt_extra,
                                  provider=args.provider)
                time.sleep(0.3)
                return a, u

            err = None
            try:
                ans, usage = _ask(ctx)
                m = FETCH_RE.match(ans or "")
                if m and args.arm == "t2_tulbase" and engine is not None:
                    fetch_id = m.group(1)
                    fetch_ok, content = engine.fetch(convs[p["conv_id"]], fetch_id)
                    block = (f"\n\n=== FETCHED BLOCK {fetch_id} ===\n{content}"
                             if fetch_ok else
                             f"\n\n=== FETCH FAILED ({fetch_id}): {content} ===")
                    ctx2 = ctx + block
                    in_chars += len(block)
                    ans, usage = _ask(ctx2)            # tek fetch turu
            except (RuntimeError, KeyError, ValueError) as e:
                # tek probe hatası TÜM koşuyu düşürmesin (gece dayanıklılığı)
                err = str(e)[:200]
                ans, usage = "", {}
                print(f"  ✗ {p['probe_id']} atlandı: {err}", flush=True)
            s = score(p, ans) if not err else 0.0
            n += 1
            total_score += s
            fh.write(json.dumps({
                "probe_id": p["probe_id"], "family": p["family"],
                "subtype": p["subtype"], "arm": args.arm,
                "provider": args.provider, "model": model,
                "score": s, "answer": ans, "usage": usage,
                "ctx_tokens": len(ctx) // 4,        # saving hesabı için
                "evidence_in_ctx": ev_cov,          # 1.0=tam, 0=hiç, None=abstain
                "extra_includes": getattr(engine, "last_extra", None),
                "fetch_id": fetch_id, "fetch_ok": fetch_ok,
                "summarizer": summ_mode, "embedder": embed_mode,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):       # canlılık göstergesi
                el = (time.time() - t_start) / 60
                eta = el / n * (len(todo) - n) if n else 0
                print(f"  [{n}/{len(todo)}] son={p['probe_id']} "
                      f"ort={total_score / n:.3f} geçen={el:.1f}dk "
                      f"kalan~{eta:.0f}dk", flush=True)

    print(f"{args.arm}/{args.provider}: {n} probes, "
          f"mean score {total_score / n:.3f}" if n else "nothing to do")
    print(f"  approx input ~{in_chars // 4} tokens -> {out_path.name}")
    # family breakdown
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r["score"])
    for f in sorted(fams):
        v = fams[f]
        print(f"  {f}: {sum(v)/len(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
