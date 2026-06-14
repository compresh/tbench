# Compresh results

Per-probe outputs from Compresh's own runs on T-bench v1.1 (answerer:
gpt-5-mini), published so you can re-score and audit them with the deterministic
scorer in `runner/score.py`, and compare your own system against them.

## Files — `v1.1/`

| file | arm | what it is |
|---|---|---|
| `raw_gpt-5-mini.jsonl` | raw | full history (control) |
| `oracle_gpt-5-mini.jsonl` | oracle | evidence turns only (upper bound) |
| `tulbase_gpt-5-mini.jsonl` | tulbase | open-source compression core |
| `tul2.0_gpt-5-mini.jsonl` | tulngin / TUL 2.0 | query-aware retrieval (in production) |
| `tul2.1_gpt-5-mini.jsonl` | TUL 2.1 | retrieval + role/anchor (on the way) |

Source runs: `tbench-008fix` (raw / oracle / tulbase / TUL 2.0) + `tbench-009`
(TUL 2.1, corrected anchor).

Each line holds `{probe_id, family, subtype, arm, score, answer, ctx_tokens,
...}`. The `score` is already the deterministic score; to re-score
independently, join each line's `answer` with the matching probe in
`../../data/v1.1/probes.json` and apply `runner/score.py`.

`raw`, `oracle`, `tulbase` are reproducible with the open harness. `tulngin`
(TUL 2.0 / 2.1) is Compresh's hosted engine — reproduce those columns against
the API at `api.compre.sh`, or audit the answers/scores published here directly.
