#!/usr/bin/env python3
"""tulv.ing X-RAY report — the C1-C7 profile (vision: diagnosis, not a score).

Reads full-grid jsonl results + controls, re-scores every answer with the
CURRENT scorer (report_grid philosophy), and emits the diagnostic profile:

  1. arm x family table (composite kept INTERNAL — never the headline)
  2. arm x C-axis table (primary-only; C6 row carries C6_gain, C2 row is
     the exposure slice)
  3. transport-vs-LLM loss split per arm (evidence_in_ctx; v0.6.2 runs
     carry it natively)
  4. bootstrap 95% CI on family accuracies (2000 resamples)
  5. token accounting (in/out means)

Usage (after tbench-015 arms finish):
  python3 xray_report.py --run-id tbench-015 --data ../data/v2.0-public
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score                                    # noqa: E402

RES = Path(__file__).parent / "results"
ARMS = ["t0_raw", "t1_oracle", "t2_tulbase", "t3_tul20", "t6_tul21p0"]
FAMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
CAXES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def load(name, golds):
    p = RES / name
    if not p.exists():
        return None
    rows = {}
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["probe_id"]] = r
    return {pid: dict(r, rescore=score(golds[pid], r["answer"] or ""))
            for pid, r in rows.items() if pid in golds}


def acc(rows, pred=lambda r: True):
    v = [r["rescore"] for r in rows.values() if pred(r)]
    return (sum(v) / len(v) if v else float("nan")), len(v)


def ci95(vals, n_boot=2000, seed=7):
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals)
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="tbench-015")
    ap.add_argument("--data", default="../data/v2.0-public")
    ap.add_argument("--provider", default="openai")
    args = ap.parse_args()

    data = Path(__file__).parent / args.data if not Path(args.data).is_absolute() \
        else Path(args.data)
    probes = {p["probe_id"]: p
              for p in json.loads((data / "probes.json").read_text())}
    ctrls = {c["probe_id"]: c
             for c in json.loads((data / "controls.json").read_text())}
    manifest = json.loads((data / "MANIFEST.json").read_text())
    exp_of_turn = {}                       # conv -> global_turn -> exposure
    for pl in manifest["plants"]:
        exp_of_turn[(pl["conv_id"], pl["global_turn"])] = \
            pl.get("exposure_count", 1)

    def multi_exposed(p):
        ev = p.get("evidence_turns") or []
        return any(exp_of_turn.get((p["conv_id"], t), 1) >= 2 for t in ev)

    arms = {a: r for a in ARMS
            if (r := load(f"{args.run_id}_{a}_{args.provider}.jsonl",
                          probes))}
    nh = load(f"{args.run_id}_no_history_{args.provider}.jsonl", ctrls)
    twinned = {r["twin_of"] for r in nh.values()} if nh else set()
    nh_acc = acc(nh)[0] if nh else float("nan")

    L = [f"# {args.run_id} — tulv.ing röntgen raporu", "",
         f"Set: {data.name} · gen {manifest['generator_version']} · seed "
         f"{manifest['seed']} · {manifest['n_probes_total']} probe · "
         "skorlar güncel scorer ile yeniden hesaplandı", ""]

    # 1 — arm x family (+CI)
    L += ["## Aile tablosu (%95 bootstrap CI)", "",
          "| kol | " + " | ".join(FAMS) + " |",
          "|---|" + "---|" * len(FAMS)]
    for a, rows in arms.items():
        cells = []
        for f in FAMS:
            v = [r["rescore"] for r in rows.values() if r["family"] == f]
            if not v:
                cells.append("—")
                continue
            lo, hi = ci95(v)
            cells.append(f"{sum(v)/len(v):.3f} [{lo:.2f}-{hi:.2f}]")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    # 2 — arm x C-axis (primary-only; the actual X-ray)
    L += ["", "## RÖNTGEN — C1-C7 profili (primary-only)", "",
          "| kol | " + " | ".join(CAXES) + " |",
          "|---|" + "---|" * len(CAXES)]
    for a, rows in arms.items():
        cells = []
        for cx in CAXES:
            if cx == "C2":
                m1, n1 = acc(rows, lambda r: not multi_exposed(probes[r["probe_id"]])
                             and probes[r["probe_id"]]["subtype"] in
                             ("confusable_integrity", "confirmed_integrity",
                              "third_party_confusable"))
                m2, n2 = acc(rows, lambda r: multi_exposed(probes[r["probe_id"]]))
                cells.append(f"Δ{(m2-m1)*100:+.1f}pp (1×{n1}/n{n2})"
                             if n2 else "—")
                continue
            v = [r["rescore"] for r in rows.values()
                 if probes[r["probe_id"]]["criteria"]["primary"] == cx]
            if not v:
                cells.append("—")
                continue
            cell = f"{sum(v)/len(v):.3f}"
            if cx == "C6" and nh and twinned:
                g, _ = acc(rows, lambda r: r["probe_id"] in twinned)
                cell += f" (gain{(g-nh_acc)*100:+.0f}pp)"
            cells.append(cell)
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    if nh:
        L.append(f"\nno_history kontrol: {nh_acc:.3f} (n={len(nh)}; "
                 "kol-bağımsız tek koşu)")
    L.append("\nC2 sütunu = multi-exposure − single-exposure kesit farkı "
             "(aynı alt-türler; rejim confound'u şerhli — DESIGN-v2 §5). "
             "Kompozit sayı bilinçli olarak YOK (vizyon: skor değil profil).")

    # 3 — transport vs LLM loss (evidence_in_ctx)
    L += ["", "## Kayıp ayrışımı (kanıt context'te miydi?)", "",
          "| kol | kanıt-var & yanlış (LLM) | kanıt-eksik & yanlış "
          "(taşıma) | acc|kanıt-tam | ort. kapsama |", "|---|---|---|---|---|"]
    for a, rows in arms.items():
        withev = [r for r in rows.values()
                  if r.get("evidence_in_ctx") is not None]
        if not withev:
            L.append(f"| {a} | — koşu-anı kaydı yok (v0.6.2 öncesi; "
                     "evidence_audit.py kullan) | | | |")
            continue
        llm = sum(1 for r in withev
                  if r["evidence_in_ctx"] >= 1 and r["rescore"] < 1)
        tra = sum(1 for r in withev
                  if r["evidence_in_ctx"] < 1 and r["rescore"] < 1)
        full = [r["rescore"] for r in withev if r["evidence_in_ctx"] >= 1]
        cov = sum(r["evidence_in_ctx"] for r in withev) / len(withev)
        if not full:          # t2: elided-md render'da turn etiketi yok —
            L.append(f"| {a} | — ölçüm uygulanamaz (etiketsiz render) | | | |")
            continue
        L.append(f"| {a} | {llm} | {tra} | "
                 f"{sum(full)/len(full):.3f} (n={len(full)}) | {cov:.2f} |")

    # 4 — tokens
    L += ["", "## Token muhasebesi (ort/istek)", "",
          "| kol | in | out |", "|---|---|---|"]
    for a, rows in arms.items():
        ins = [r["usage"].get("prompt_tokens") for r in rows.values()
               if r.get("usage", {}).get("prompt_tokens") is not None]
        outs = [r["usage"].get("completion_tokens") for r in rows.values()
                if r.get("usage", {}).get("completion_tokens") is not None]
        L.append(f"| {a} | {sum(ins)/len(ins):,.0f} | "
                 f"{sum(outs)/len(outs):,.0f} |"
                 if ins and outs else f"| {a} | — | — |")

    out = "\n".join(L)
    print(out)
    rep = Path(__file__).parent.parent / "reports"
    rep.mkdir(exist_ok=True)
    (rep / f"{args.run_id}_xray.md").write_text(out + "\n")
    print(f"\n-> reports/{args.run_id}_xray.md", file=sys.stderr)


if __name__ == "__main__":
    main()
