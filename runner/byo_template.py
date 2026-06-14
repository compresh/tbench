"""
Template — benchmark YOUR memory/context system on T-bench.

Copy this file, implement build_context(), then run:

  python run_tbench.py --data ../data/v1.1 --arm byo --byo your_file.py \\
      --provider openai --model gpt-5-mini

You return the context string your system would feed the model for each probe.
The runner then asks the model your context + the probe question, and scores the
answer deterministically. The weaker your model, the more your memory layer has
to carry — that is the point of the test.
"""


def build_context(conversation: dict, probe: dict) -> str:
    """Return the context YOUR system would supply for this probe.

    conversation = {
        "conv_id": str, "regime": "S"|"M"|"L", "n_turns": int,
        "sessions": [ { "label": str, "day": "YYYY-MM-DD",
                        "turns": [ {"turn": int, "role": "user"|"assistant",
                                    "text": str}, ... ] }, ... ]
    }
    probe = {"probe_id", "conv_id", "family", "subtype", "question", ...}

    Only `probe["question"]` is the live query. Everything in `conversation` is
    the history your system gets to compress / retrieve / remember.

    FAIRNESS: do NOT read `probe["gold"]` or `probe["evidence_turns"]` — those
    are the answer key. Using them is cheating and the result is meaningless.
    """
    # Trivial example: feed the whole history (equivalent to the `raw` arm).
    # Replace this with your retrieval / compression / memory logic.
    lines = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            lines.append(f"[turn {turn['turn']} | {turn['role']}] {turn['text']}")
    return "\n".join(lines)
