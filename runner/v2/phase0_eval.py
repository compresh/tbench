#!/usr/bin/env python3
"""tbench-013 (TUL 2.1 Faz 0) gate evaluator — pre-registered thresholds.

Compares the t6_tul21p0 run against tbench-012 reference arms on a
CONSISTENT scoring basis: every recorded answer is re-scored with the
CURRENT probes.json/controls.json golds (report_grid philosophy), so the
widened P6a lexicon applies to all arms equally.

  python3 phase0_eval.py                     # defaults: 013 vs 012, openai
  python3 phase0_eval.py --run-id tbench-013 --ref-run tbench-012

Gates (pre-registered in RUNS.md tbench-013 BEFORE the run):
  E1  acc(t6, F6)  >=  acc(t0@012, F6)      same-gold basis (kira kanıtı)
  E2  C6_gain(t6)  >=  +10pp                gain = grounded(twinned) - no_history
  E3  mean ctx(t6) <   mean ctx(t0) * 0.5   keep-all dejenerasyon koruması
  E4  acc(t6, F7)  >=  0.95                 yan hasar yok (012'de tavandı)
Bilgi satırları: t3 kıyası (mekanizma makası), extra_includes ortalaması.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score                                   # noqa: E402

RES = Path(__file__).parent / "results"


def load_rescored(name: str, golds: dict) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"eksik: {p.name}")
    rows = {}
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["probe_id"]] = r
    return {pid: dict(r, rescore=score(golds[pid], r["answer"] or ""))
            for pid, r in rows.items()}


def acc(rows, pred=lambda r: True):
    v = [r["rescore"] for r in rows.values() if pred(r)]
    return (sum(v) / len(v) if v else float("nan")), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="tbench-013")
    ap.add_argument("--ref-run", default="tbench-012")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--data", default="../data/v2-pilot",
                    help="probe/control golds (tbench-014: ../data/v2-pilot-ood)")
    args = ap.parse_args()

    data = Path(__file__).parent / args.data if not Path(args.data).is_absolute() \
        else Path(args.data)
    probes = {p["probe_id"]: p
              for p in json.loads((data / "probes.json").read_text())}
    ctrls = {c["probe_id"]: c
             for c in json.loads((data / "controls.json").read_text())}

    t6 = load_rescored(f"{args.run_id}_t6_tul21p0_{args.provider}.jsonl", probes)
    t0 = load_rescored(f"{args.ref_run}_t0_raw_{args.provider}.jsonl", probes)
    t3 = load_rescored(f"{args.ref_run}_t3_tul20_{args.provider}.jsonl", probes)
    nh = load_rescored(f"{args.ref_run}_no_history_{args.provider}.jsonl", ctrls)

    twinned = {r["twin_of"] for r in nh.values()}
    nh_acc, _ = acc(nh)

    print(f"== tbench-013 Faz 0 gates (zemin: güncel gold'larla re-score) ==")
    for name, rows in (("t6_tul21p0", t6), ("t0@ref", t0), ("t3@ref", t3)):
        f6, n6 = acc(rows, lambda r: r["family"] == "F6")
        f7, n7 = acc(rows, lambda r: r["family"] == "F7")
        cx = [r.get("ctx_tokens") or 0 for r in rows.values()]
        print(f"  {name:11s} F6={f6:.3f} (n={n6})  F7={f7:.3f} (n={n7})  "
              f"ctx_ort={sum(cx)/len(cx):,.0f}")
    print(f"  no_history  acc={nh_acc:.3f} (n={len(nh)})")

    t6f6, _ = acc(t6, lambda r: r["family"] == "F6")
    t0f6, _ = acc(t0, lambda r: r["family"] == "F6")
    t3f6, _ = acc(t3, lambda r: r["family"] == "F6")
    t6f7, _ = acc(t6, lambda r: r["family"] == "F7")
    g6, _ = acc(t6, lambda r: r["probe_id"] in twinned)
    gain = g6 - nh_acc
    cx6 = [r.get("ctx_tokens") or 0 for r in t6.values()]
    cx0 = [r.get("ctx_tokens") or 0 for r in t0.values()]
    m6, m0 = sum(cx6) / len(cx6), sum(cx0) / len(cx0)
    extra = [r["extra_includes"] for r in t6.values()
             if r.get("extra_includes") is not None]

    print(f"\nE1 kira kanıtı : t6 F6 {t6f6:.3f} >= t0 F6 {t0f6:.3f} ->",
          "GEÇTİ" if t6f6 >= t0f6 else "TAKILDI")
    print(f"E2 C6_gain     : grounded {g6:.3f} - nh {nh_acc:.3f} = {gain:+.3f}"
          f" >= +0.100 ->", "GEÇTİ" if gain >= 0.100 else "TAKILDI")
    print(f"E3 dejenerasyon: ctx {m6:,.0f} < t0 {m0:,.0f} x 0.5 ->",
          "GEÇTİ" if m6 < m0 * 0.5 else "TAKILDI (keep-all'a mı düştük?)")
    print(f"E4 yan hasar   : t6 F7 {t6f7:.3f} >= 0.95 ->",
          "GEÇTİ" if t6f7 >= 0.95 else "TAKILDI")
    print(f"\nBilgi: mekanizma makası t6-t3 F6 = {(t6f6 - t3f6) * 100:+.1f}pp"
          f" · extra_includes ort = "
          f"{(sum(extra) / len(extra)) if extra else float('nan'):.1f}"
          f" turn/istek · gürültü bandı ±1 probe")


if __name__ == "__main__":
    main()
