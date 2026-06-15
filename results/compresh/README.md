# Compresh results

Per-probe outputs from Compresh's own runs on T-bench v1.1 (answerer:
gpt-5-mini), published so you can re-score and audit them with the deterministic
scorer in `runner/score.py`, and compare your own system against them.

## Files — `v1.1/`

| file | arm | what it is |
|---|---|---|
| `raw_gpt-5-mini.jsonl` | raw | full history (control) |
| `naive_rag_gpt-5-mini.jsonl` | naive RAG (baseline) | policy-free embedding top-k |
| `oracle_gpt-5-mini.jsonl` | oracle | evidence turns only (upper bound) |
| `tulbase_gpt-5-mini.jsonl` | tulbase | open-source compression core |
| `tul2.0_gpt-5-mini.jsonl` | retrieval (TUL 2.0) | query-aware retrieval (in production) |
| `tul2.1_gpt-5-mini.jsonl` | retrieval + role/anchor (TUL 2.1) | adds encode-time anchor (on the way) |

Source runs: `tbench-008fix` (raw / oracle / tulbase / TUL 2.0) + `tbench-009`
(TUL 2.1, corrected anchor) + the bundled `naive_rag` arm (gpt-5-mini).

Each line holds `{probe_id, family, subtype, arm, score, answer, ctx_tokens,
...}`. The `score` is already the deterministic score; to re-score
independently, join each line's `answer` with the matching probe in
`../../data/v1.1/probes.json` and apply `runner/score.py`.

`raw`, `naive RAG`, and `oracle` reproduce directly with the bundled harness;
`tulbase` is the open-source compression core (reproducible via the `tulbase`
package). The `TUL 2.0 / 2.1` columns run on Compresh's hosted system
(tulngin = tulbase + TUL 2.0) — reproduce them against the API at
`api.compre.sh`, or audit the answers and scores published here directly.
