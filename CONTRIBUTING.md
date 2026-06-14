# Contributing to T-bench

T-bench is a diagnostic instrument, not a leaderboard. The most valuable
contribution is **your system's results** — we want to see where your memory
stack's X-ray differs from ours.

## Share + discuss

Open a [Discussion](https://github.com/compresh/tbench/discussions) to post what
you found — a family your system fails, a probe you think is unfair, a surprising
result. Discussion is the default; it is collaborative, not competitive.

## Submit results (pull request)

1. Run the harness on your system (see `runner/README.md`).
2. Add your output under `results/<your-handle>/<system>.jsonl`.
3. Include a short `results/<your-handle>/README.md`: the system, the answerer
   model + context limit, the dataset version, and how to reproduce.
4. Open a PR. We do not re-run or re-configure your system — you run it, the
   deterministic harness scores it, the git history records the provenance.

We will not silently "tune" a contributed baseline to look worse or better. If a
result looks off, we discuss it in the open.

## Report an issue with the benchmark itself

A leaky probe, an ambiguous question, a scorer false-positive — open an
[Issue](https://github.com/compresh/tbench/issues). We hold ourselves to the
same standard: pre-registered thresholds, published negatives, fixes logged.

## What we will not do

Build a competitive ranking. T-bench exists so people can diagnose their own
systems against episodic-memory criteria — not to crown a winner.
