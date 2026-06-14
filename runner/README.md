# T-bench runner

The harness: load a dataset, run an arm, score deterministically, write JSONL.

```bash
pip install -r requirements.txt
python run_tbench.py --data ../data/v1.1 --arm your_system --out results/you.jsonl
python report_grid.py results/you.jsonl          # family breakdown + overall
```

## Arms

Built-in reference arms: `raw` (full context), `oracle` (evidence turns only),
`blind_summarization`, `retrieval`. To score your own system, implement an arm
in `engine_arms.py` that takes the conversation history + probe and returns the
context your system would feed the answerer (or the answer directly).

## Scoring

Deterministic, derived from the generating graph (substring / structural / exact
match). An LLM judge is used **only** for one free-text sub-probe (F5a), pinned
to a fixed model. Every run records the engine commit, config, dataset version,
answerer model + context limit, seed, summarizer, and embedder — a run with a
silent fallback (e.g. embedder degraded to a no-op) is rejected, not scored.

## Provenance

This harness mirrors the lab source of truth in Compresh's research tree; the
public copy here is the frozen, reproducible version that backs the published
report. Pre-registered hypotheses and thresholds live in the run registry.
