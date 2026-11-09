"""
build_dataset.py — Filter winning traces → SFT training pairs.

Reads all JSONL from traces/, keeps injective non-known-rule episodes,
reconstructs the LLM prompt from the saved state, and writes
(prompt, completion) pairs to dataset/train.jsonl.

Usage:
    python3 build_dataset.py
    python3 build_dataset.py --include-known   # also include known-rule successes
    python3 build_dataset.py --min-coverage 0.8  # partial successes too
"""

import argparse
import glob
import json
import os

from llm_agent import build_prompt

KNOWN_RULES = {"union_min", "singleton_injection"}
DATASET_DIR = "dataset"
OUT_PATH    = f"{DATASET_DIR}/train.jsonl"


def load_all_traces(pattern="traces/*.jsonl") -> list[dict]:
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def rebuild_prompt(r: dict) -> str:
    """Reconstruct the LLM prompt from a saved trace record."""
    family = [frozenset(F) for F in r["family"]]
    return build_prompt(family, r["n"], r["phi_scores"], r["freqs"], r["episode"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-known",  action="store_true",
                        help="Include known-rule (union_min, singleton) successes")
    parser.add_argument("--min-coverage",   type=float, default=1.0,
                        help="Minimum φ-coverage fraction (default=1.0: injective only)")
    args = parser.parse_args()

    records = load_all_traces()
    if not records:
        print("No trace records found — run collect_traces.py first.")
        return
    print(f"Loaded {len(records)} trace records from traces/")

    # ── Filter ────────────────────────────────────────────────────────────────
    winners = [
        r for r in records
        if r.get("injective")
        and not r.get("parse_fail")
        and r.get("coverage", 0.0) >= args.min_coverage
        and r.get("mapping_trace")   # must have full φ recorded
        and r.get("family")          # must have family for prompt rebuild
    ]

    if not args.include_known:
        winners = [r for r in winners if r.get("rule") not in KNOWN_RULES]

    print(f"Kept {len(winners)} winning records "
          f"({'including' if args.include_known else 'excluding'} known rules, "
          f"min-coverage={args.min_coverage:.0%})")

    if not winners:
        print("No winners after filtering. Try --min-coverage 0.8 or --include-known.")
        return

    # ── Stats breakdown ───────────────────────────────────────────────────────
    by_n    = {}
    by_rule = {}
    for r in winners:
        by_n[r["n"]]       = by_n.get(r["n"], 0) + 1
        by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1

    print("\nBreakdown by n:")
    for n, cnt in sorted(by_n.items()):
        print(f"  n={n}: {cnt} pairs")
    print("Breakdown by rule:")
    for rule, cnt in sorted(by_rule.items(), key=lambda x: -x[1]):
        flag = "  ← known" if rule in KNOWN_RULES else ""
        print(f"  {rule:<25}: {cnt}{flag}")

    # ── Write dataset ─────────────────────────────────────────────────────────
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in winners:
            prompt     = rebuild_prompt(r)
            completion = json.dumps({
                "element": r["element"],        # 1-indexed
                "rule":    r["rule_idx"],        # 0-indexed
                "reason":  r.get("reason", ""),
            })
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

    print(f"\nWrote {len(winners)} SFT pairs → {OUT_PATH}")
    print("Next step: python3 finetune.py")


if __name__ == "__main__":
    main()
