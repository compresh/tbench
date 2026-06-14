# Results

A shared corpus of T-bench results — **not a leaderboard.** Each contributor's
runs live under their own folder; the git history is the provenance.

## Layout

```
results/
  compresh/        our own arms (raw, oracle, blind_summarization, retrieval, …)
  <your-handle>/   your system's runs
    <system>.jsonl
    README.md      system, answerer model + context limit, dataset version, reproduce steps
```

## How results land here

The harness writes JSONL here (`run_tbench.py --out`). To contribute, open a PR
adding your folder (see [../CONTRIBUTING.md](../CONTRIBUTING.md)). We review for
provenance, not for who wins, and we do not re-tune contributed systems.

## Reading them

```bash
python ../runner/report_grid.py results/<handle>/<system>.jsonl
```

Every probe records the context fed, the answer, and the deterministic score —
so a result is an X-ray you can inspect, not just a number.
