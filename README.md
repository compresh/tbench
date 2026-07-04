# T-bench

**An episodic-memory diagnostic for LLM context systems.** Point it at your own
memory/retrieval stack and see where it forgets a source, misses a correction,
or loses the thread. Report and leaderboard-free results live at
**[tulv.ing](https://tulv.ing)**.

> T-bench is an X-ray, not a photo-finish. It is built to diagnose systems —
> including ours — not to crown a winner. The dataset, the harness, and the
> report are all in this one repository. Fork it and run it on your system.

**Conflict of interest, stated up front:** T-bench is built by
[Compresh](https://compre.sh). Our own systems appear under the same conditions
as every other arm, and we publish a pre-registered hypothesis about our newest
system that *failed* (see [Honesty](#honesty)). Hiding that would defeat the point.

## v2.0 — the first full X-ray (July 2026)

**Report: [tulv.ing/report-v2.html](https://tulv.ing/report-v2.html).**
1,264 probes across seven criteria (C1 binding · C2 single-exposure · C3
instance discrimination · C4 temporal · C5 source monitoring · C6 future
projection · C7 consolidation), planted in ~1.17M-token multi-session
haystacks with confusable twins, OOD phrasings and decoy sentences.
Headlines: similarity retrieval (our product included) has a structural
C6 blind spot (0.52 vs raw 0.94, replicated across three sets and two
seeds); the cause is selection, not compression; with evidence delivered,
model accuracy is >=0.99 — and a Phase-0 dual-channel candidate recovers
the blind spot to 0.88 with zero regression, on ~3% of raw's tokens.
Two pre-registered expectations failed and are published (C7 at ceiling;
un-stamped oracle evidence loses temporal questions).

**Data:** [`data/v2.0-public/`](data/v2.0-public/) — frozen 2026-07-04,
generator 0.7.0, seed 404: `conversations.json`, `probes.json` (1,264),
`controls.json` (80 no-history twins), `MANIFEST.json` (criteria map,
exposure histogram, plants). A private held-out split exists; its seed is
stored nowhere digital. Anchored rebuilds keep versions comparable
(v2.0 reproduced the v1.1 composite exactly: 0.988 = 0.988).

**Reproduce (model arms):**
```bash
cd runner/v2
python3 run_tbench.py --data ../../data/v2.0-public --arm t0_raw \
    --provider openai --run-id myrun          # raw upper line
python3 run_tbench.py --data ../../data/v2.0-public --arm t1_oracle \
    --provider openai --run-id myrun          # evidence-only floor
python3 run_tbench.py --data ../../data/v2.0-public --controls \
    --provider openai --run-id myrun          # no-history twins
python3 xray_report.py --run-id myrun --data ../../data/v2.0-public
```
Scoring is deterministic (`score.py --selftest`). The Compresh engine
arms (tulbase / TUL 2.0 / Phase-0) depend on our bench-local engine and
are our published rows; to X-ray **your** system, wrap it as a context
builder per `runner/byo_template.py` and run the same probes. Every
result row records `evidence_in_ctx`, so you get the transport-vs-model
loss split for free.

## TL;DR

The value of a context architecture is **inversely proportional to the strength
of the answering model.**

- On a strong model, retrieval matches full-context quality at **~99% fewer
  tokens** — the win is cost, not accuracy.
- On a small open model, the same retrieval lifts accuracy by **~22 points**.
- On the cheapest models, the raw long history **does not even fit** — retrieval
  is the only way the task runs at all.

## Main grid — strong answerer (gpt-5-mini, v1.1, 976 probes)

| Arm | Overall | F1 src | F2 corr | F3 time | F4 bind | F5 calib | ctx (tok) | saving |
|---|---|---|---|---|---|---|---|---|
| raw (control) | 0.955 | 1.000 | 1.000 | 0.828 | 0.932 | 0.983 | 31,947 | — |
| naive RAG (baseline) | 0.900 | 0.964 | 1.000 | 0.667 | 0.792 | 0.975 | 270 | 99.2% |
| oracle (upper bound) | 0.985 | 0.951 | 1.000 | 0.984 | 0.990 | 1.000 | 35 | 99.9% |
| blind-summarization | 0.991 | 0.991 | 1.000 | 1.000 | 0.958 | 0.988 | 38,061 | −19.1% |
| retrieval (TUL 2.0) | 0.988 | 0.973 | 1.000 | 1.000 | 0.969 | 0.988 | 275 | 99.1% |
| retrieval + role/anchor (TUL 2.1) | 0.990 | 0.982 | 1.000 | 1.000 | 0.969 | 0.988 | 297 | 99.1% |

`raw` quality falls with length (S 0.995 → L 0.917); retrieval stays flat (~0.99).
At retrieval's budget (~270 tok), naive RAG scores only 0.900 — below `raw` — losing
on the temporal and binding families (F3 0.667, F4 0.792).
Full analysis, the weak-model curve, and the methodology: **[tulv.ing](https://tulv.ing)**.

## Run it on your own system

```bash
git clone https://github.com/compresh/tbench
cd tbench/runner
pip install -r requirements.txt
python run_tbench.py --data ../data/v1.1 --arm your_system --out results/you.jsonl
```

Scoring is deterministic (an LLM judge is used for one free-text sub-probe only),
so your numbers are reproducible. Then:

- **Share + discuss** what you found in [Discussions](https://github.com/compresh/tbench/discussions).
- **Contribute results** by opening a PR that adds your `.jsonl` to `results/`
  (see [CONTRIBUTING](CONTRIBUTING.md)). This is collaborative, not competitive —
  we want to see where *your* system's X-ray differs from ours.

T-bench is a **living benchmark** — we keep adding tests, models, and harder probes
over time, and we are glad to publish results, ours and the community's.

## What it measures

Functional probes inspired by Tulving's episodic-memory framework (not a claim
that any system "satisfies Tulving's criteria" — we do not measure autonoesis):

| Family | Probes |
|---|---|
| F1 — source attribution | who said it: user, assistant, or third party; which turn |
| F2 — correction fidelity | the current value *and* recall of the superseded one |
| F3 — relational time | ordering, dangling-reference resolution |
| F4 — what-where-when binding | episode individuation, co-occurrence, counting |
| F5 — remember/know | calibration: hearsay flags, abstention, false-memory rejection |

## How to trust the numbers

- **Pre-registered** hypotheses and thresholds, written before each run in an
  append-only registry. Thresholds are never moved after seeing results, and
  negative results are published.
- **Deterministic scoring** from a generating graph — reproducibility does not
  depend on a judge model.
- **Synthetic, graph-based** generation → contamination is controllable (public
  split + private held-out split, regenerable).

## Honesty

A pre-registered hypothesis (H1) predicted our newest system would beat plain
retrieval by ≥ +5pp on source attribution. Measured: **+0.9pp. Rejected.** The
honest read is that role/anchor rendering is a net-positive, zero-cost change —
decisive on weak models, marginal on strong ones — not the across-the-board win
we guessed. T-bench has also caught several of our own bugs (a mis-firing
temporal anchor; two silent fallbacks) before they shipped. An instrument that
never catches its builder is not an instrument.

## Repository layout

```
tbench/
├── docs/        the tulv.ing report page (GitHub Pages, served from /docs)
├── data/        T-bench dataset (versioned; public + private held-out)
├── runner/      the harness — pip install + one command
├── results/     our results + community-contributed results (via PR)
└── CONTRIBUTING.md
```

## License

Code: MIT (see `LICENSE`), aligned with other Compresh repos. Dataset: published
as measured, **CC BY-SA 4.0** with attribution — it includes StackExchange-derived
distractor text. See `ATTRIBUTION.md` and `data/README.md`.

---

*Built by [Compresh](https://compre.sh). Report: [tulv.ing](https://tulv.ing).
We are early in this area too — if T-bench tells you something we got wrong,
that is the instrument working.*
