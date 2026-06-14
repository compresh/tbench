#!/usr/bin/env python3
"""
T-bench deterministic scorer.

Scoring types (see the report at tulv.ing):
  contains             gold.contains_any   -> 1.0 if any present
  source_match         gold.source         -> rule-based source classification
  set_f1               gold.gold_set       -> recall fraction
  contains_all_groups  gold.groups         -> every group needs >=1 hit
  abstain              gold.abstain_markers (+ forbid_digits)

All matching is case-insensitive. `source_match` is best-effort deterministic.
Run `python score.py --selftest` before trusting any change to this file —
the numbers in the report depend on it being byte-stable.
"""

from __future__ import annotations
import re
import sys


def _norm(s: str) -> str:
    # Fold typographic punctuation to ASCII: models write "didn't" with a
    # curly apostrophe (U+2019) while the patterns below look for a straight one.
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("‑", "-").replace("–", "-"))


def _has(text: str, needle: str) -> bool:
    # Short numeric golds ("4", "15") need word boundaries so "42" does not
    # satisfy the gold "4".
    text = _norm(text)
    if len(needle) <= 2 and needle.isdigit():
        return re.search(rf"\b{re.escape(needle)}\b", text) is not None
    return needle.lower() in text.lower()


def score_contains(answer: str, gold: dict) -> float:
    return 1.0 if any(_has(answer, g) for g in gold["contains_any"]) else 0.0


def score_set_f1(answer: str, gold: dict) -> float:
    items = gold["gold_set"]
    found = sum(1 for g in items if _has(answer, g))
    return found / len(items) if items else 0.0


def score_groups(answer: str, gold: dict) -> float:
    ok = all(any(_has(answer, g) for g in group) for group in gold["groups"])
    return 1.0 if ok else 0.0


def score_abstain(answer: str, gold: dict) -> float:
    if gold.get("forbid_digits") and re.search(r"\d", answer):
        return 0.0
    return 1.0 if any(_has(answer, m) for m in gold["abstain_markers"]) else 0.0


_ASSISTANT_PAT = re.compile(
    r"\bI\b.{0,40}\b(work|calculat|deriv|figur|infer|comput|added|put"
    r"|multipl|combin|total|summ?ed)\w*",
    re.IGNORECASE | re.DOTALL)
_USER_PAT = re.compile(
    r"\b(you|yourself)\b.{0,40}\b(said|mention|told|stated|brought|first"
    r"|wrote)\w*"
    r"|\bit was you\b|\byou did\b|\byou first\b"
    r"|\byour (message|note|update|words)\b",
    re.IGNORECASE | re.DOTALL)


def score_source(answer: str, gold: dict) -> float:
    src = gold["source"]
    if src.startswith("third:"):
        name = src.split(":", 1)[1]
        return 1.0 if _has(answer, name) else 0.0
    third_named = any(_has(answer, n) for n in
                      ("Sarah", "Tom", "Priya"))           # known third parties
    if src == "assistant":
        return 1.0 if (_ASSISTANT_PAT.search(answer) and not third_named) else 0.0
    if src == "user":
        return 1.0 if (_USER_PAT.search(answer) and not third_named) else 0.0
    return 0.0


SCORERS = {
    "contains": score_contains,
    "set_f1": score_set_f1,
    "contains_all_groups": score_groups,
    "abstain": score_abstain,
    "source_match": score_source,
}


def score(probe: dict, answer: str) -> float:
    return SCORERS[probe["scoring"]](answer or "", probe["gold"])


# ------------------------------------------------------------------ selftest

CASES = [
    # (scoring, gold, answer, expected)
    ("contains", {"contains_any": ["Tuesday"]},
     "It's on Tuesday, you corrected it earlier.", 1.0),
    ("contains", {"contains_any": ["Tuesday"]},
     "Monday, as you said at the start.", 0.0),
    ("set_f1", {"gold_set": ["pet insurance", "Dr. Kim"]},
     "You mentioned the new vet Dr. Kim and looking at pet insurance.", 1.0),
    ("set_f1", {"gold_set": ["pet insurance", "Dr. Kim"]},
     "You talked about pet insurance.", 0.5),
    ("contains_all_groups",
     {"groups": [["fifth"], ["heard", "apparently", "not confirmed"]]},
     "Apparently it's moving to the fifth floor, though that's unconfirmed.", 1.0),
    ("contains_all_groups",
     {"groups": [["fifth"], ["heard", "apparently", "not confirmed"]]},
     "Yes, the office is moving to the fifth floor.", 0.0),       # no hedge
    ("abstain", {"abstain_markers": ["never mentioned", "didn't say"],
                 "forbid_digits": True},
     "You never mentioned a salary figure.", 1.0),
    ("abstain", {"abstain_markers": ["never mentioned"], "forbid_digits": True},
     "You said it was 85000.", 0.0),                              # hallucination
    ("source_match", {"source": "third:Sarah"},
     "You heard that from Sarah in accounting.", 1.0),
    ("source_match", {"source": "user"},
     "You mentioned it yourself early in the conversation.", 1.0),
    ("source_match", {"source": "assistant"},
     "I worked that out from the two figures you gave me.", 1.0),
    ("source_match", {"source": "assistant"},
     "Sarah told you that.", 0.0),
    ("contains", {"contains_any": ["4", "four"]},
     "I am not sure, possibly 42 or Thursday.", 0.0),   # boundary check
    ("contains", {"contains_any": ["4", "four"]},
     "We talked about it 4 separate times.", 1.0),
    ("source_match", {"source": "assistant"},
     "You didn't give $420 directly — I multiplied your two figures.", 1.0),
    ("source_match", {"source": "user"},
     "You — your message at T14: 'the Q3 budget got cut'.", 1.0),
    ("abstain", {"abstain_markers": ["didn't mention"], "forbid_digits": True},
     "You didn’t mention any salary figure earlier.", 1.0),   # curly apostrophe
    ("source_match", {"source": "user"},
     "You — you wrote “the Q3 budget got cut”.", 1.0),
]


def selftest() -> int:
    bad = 0
    for scoring, gold, answer, want in CASES:
        got = SCORERS[scoring](answer, gold)
        if abs(got - want) > 1e-9:
            bad += 1
            print(f"FAIL [{scoring}] want={want} got={got} :: {answer!r}")
    print(f"selftest: {len(CASES) - bad}/{len(CASES)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
