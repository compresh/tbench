#!/usr/bin/env python3
"""
T-bench arms — the open reference arms plus the bring-your-own-system hook.

An "arm" is a callable `arm(conversation, probe) -> str` that returns the
context string fed to the answerer model for one probe. The answerer then sees
only that context plus the probe's question, and is scored deterministically.

Open arms shipped here:
  raw        full conversation history (control / upper cost bound)
  oracle     only the evidence turns (quality upper bound; uses the answer key,
             so it is a reference ceiling, not a real system)
  naive_rag  vanilla embedding retrieval (MiniLM top-k) — the standard baseline

To benchmark YOUR system, copy `byo_template.py`, implement `build_context`,
and run with `--arm byo --byo your_file.py`.

Compresh's own tulngin (TUL 2.0/2.1) is not in this repo; our per-probe outputs
are published under `results/compresh/` so you can re-score and audit them with
the deterministic scorer, and you can reproduce them against the hosted API.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path


def render_turns(sessions, keep=None) -> str:
    """Flatten sessions to 'turn lines'; keep=None -> all, else a set of turn
    numbers to include."""
    lines = []
    for s in sessions:
        for t in s["turns"]:
            if keep is None or t["turn"] in keep:
                lines.append(f"[turn {t['turn']} | {t['role']}] {t['text']}")
    return "\n".join(lines)


def arm_raw(conv: dict, probe: dict) -> str:
    return render_turns(conv["sessions"])


def arm_oracle(conv: dict, probe: dict) -> str:
    # Upper bound: feed only the turns the generator marked as evidence.
    return render_turns(conv["sessions"], keep=set(probe["evidence_turns"]))


class NaiveRAG:
    """Vanilla embedding retrieval baseline: embed every turn with a
    sentence-transformer, embed the question, keep the top-k by cosine, render
    them in chronological order. No reranking, no compression, no policy."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 8):
        from sentence_transformers import SentenceTransformer  # lazy: heavy dep
        self.model = SentenceTransformer(model)
        self.top_k = top_k
        self._cache: dict[str, object] = {}

    @staticmethod
    def _lines(conv: dict) -> list[str]:
        out = []
        for s in conv["sessions"]:
            for t in s["turns"]:
                out.append(f"[turn {t['turn']} | {t['role']}] {t['text']}")
        return out

    def __call__(self, conv: dict, probe: dict) -> str:
        import numpy as np
        lines = self._lines(conv)
        cid = conv["conv_id"]
        if cid not in self._cache:
            self._cache[cid] = self.model.encode(
                lines, normalize_embeddings=True, show_progress_bar=False)
        emb = self._cache[cid]
        q = self.model.encode([probe["question"]], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        sims = np.asarray(emb) @ np.asarray(q)
        n = len(lines)
        top = sorted(range(n), key=lambda i: float(sims[i]), reverse=True)
        keep = sorted(top[:min(self.top_k, n)])
        return "\n".join(lines[i] for i in keep)


def load_byo(path: str):
    """Load a user file that defines build_context(conversation, probe) -> str."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--byo file not found: {path}")
    spec = importlib.util.spec_from_file_location("byo_system", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fn = getattr(mod, "build_context", None)
    if not callable(fn):
        raise SystemExit(
            f"{path} must define build_context(conversation, probe) -> str")
    return fn
