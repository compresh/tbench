# T-bench dataset

Synthetic, graph-based conversations with deterministic ground truth. Each
conversation is rendered from an event graph (entities, facts, provenance,
correction edges, relational-time edges, episode boundaries), so scoring is
structural and reproducible.

## Versions & splits

- `v1.1/` — 976 probes (F1:224 · F2:224 · F3:192 · F4:96 · F5:240), haystack
  ~1.15M tokens, three size regimes (S/M/L).
- **Public split** — shipped here, for development and reproduction. The numbers
  in the report are computed on exactly this set, so a download reproduces them.
- **Private held-out split** — hidden seed, regenerable, for contamination control.

## License

Dual-licensed: the **code** is MIT; the **dataset** is **CC BY-SA 4.0**. The
conversations are synthetic and ours, but the distractor turns include text
derived from public StackExchange threads (CC BY-SA), so the dataset as a whole
carries attribution + share-alike. See [`../ATTRIBUTION.md`](../ATTRIBUTION.md).

> A future release will swap in template-only distractors — fully ours, no
> share-alike — alongside the next benchmark cycle (TUL 2.1). Until then we
> publish the set exactly as measured, so the reported numbers reproduce.

## Generation

The generator (`make_tbench_dataset.py`) and its rule-based validator — which
confirms that correction phrasing, hearsay markers, and relational-time cues
survive paraphrase — reproduce the dataset from a fixed seed.
