"""
Inspect agent traces to mine novel mapping patterns.

Usage:
    python3 inspect_traces.py                        # all traces in traces/
    python3 inspect_traces.py --file traces/foo.jsonl
    python3 inspect_traces.py --successes-only       # only injective proofs
    python3 inspect_traces.py --novel-only           # exclude known rules

Each trace record contains the full mapping φ:
  mapping_trace: [{A, B, image=A∪B}, ...]

This script surfaces:
  1. Per-rule success rates
  2. Novel successful mappings with their structural patterns
     (what B was chosen relative to A — potential new rules)
  3. The families where the agent struggled (low coverage, not injective)
     — these are candidates for adding harder test cases
"""

import argparse
import glob
import json
from collections import defaultdict


KNOWN_RULES = {"union_min", "singleton_injection"}


def load_traces(path_glob: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(path_glob)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def describe_mapping(trace: list[dict]) -> str:
    """
    Summarise the structural pattern of a mapping φ.
    For each (A, B, image) triple, describe how B relates to A.
    """
    patterns = []
    for t in trace:
        A, B, img = set(t["A"]), set(t["B"]), set(t["image"])
        if set(A) <= set(B):
            rel = "B⊇A"
        elif not (set(A) & set(B)):
            rel = "B∩A=∅"
        elif set(B) <= set(A):
            rel = "B⊆A"
        else:
            rel = "B∩A≠∅"
        patterns.append(rel)
    counts = {}
    for p in patterns:
        counts[p] = counts.get(p, 0) + 1
    return "  ".join(f"{p}×{c}" for p, c in sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",           default="traces/*.jsonl")
    parser.add_argument("--successes-only", action="store_true")
    parser.add_argument("--novel-only",     action="store_true")
    args = parser.parse_args()

    records = load_traces(args.file)
    if not records:
        print("No trace records found.")
        return

    if args.successes_only:
        records = [r for r in records if r["injective"]]
    if args.novel_only:
        records = [r for r in records if r["rule"] not in KNOWN_RULES]

    print(f"Loaded {len(records)} trace records\n")

    # ── 1. Per-rule summary ───────────────────────────────────────────────────
    rule_total   = defaultdict(int)
    rule_success = defaultdict(int)
    for r in records:
        rule_total[r["rule"]] += 1
        if r["injective"]:
            rule_success[r["rule"]] += 1

    print("── Rule success rates ────────────────────────────────────────")
    print(f"  {'Rule':<25} {'Total':>7} {'Proved':>7} {'Rate':>7}")
    print("  " + "-" * 50)
    for rule, total in sorted(rule_total.items(), key=lambda x: -x[1]):
        s    = rule_success[rule]
        flag = "  ← known" if rule in KNOWN_RULES else ""
        print(f"  {rule:<25} {total:>7} {s:>7} {s/total:>7.1%}{flag}")

    # ── 2. Novel successful mappings + their structural pattern ───────────────
    novel_success = [r for r in records
                     if r["injective"] and r["rule"] not in KNOWN_RULES
                     and r.get("mapping_trace")]

    print(f"\n── Novel successful mappings ({len(novel_success)} records) ─────")
    pattern_counts = defaultdict(int)
    for r in novel_success:
        pat = describe_mapping(r["mapping_trace"])
        pattern_counts[pat] += 1

    print("  Structural patterns of B relative to A in φ(A) = A∪B:")
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>4}×  {pat}")

    # ── 3. Show example novel mapping in detail ───────────────────────────────
    if novel_success:
        ex = novel_success[0]
        print(f"\n── Example novel proof (ep {ex['episode']}, rule={ex['rule']}) ─")
        sets_str = ", ".join(
            "{" + ",".join(str(e+1) for e in sorted(F)) + "}" if F else "∅"
            for F in ex["family"]
        )
        print(f"  Family : {sets_str}")
        print(f"  Element: x{ex['element']}")
        print(f"  F_bad  : {ex['F_bad']}")
        print(f"  F_good : {ex['F_good']}")
        print(f"  Mapping φ:")
        for t in ex["mapping_trace"]:
            A   = "{" + ",".join(str(e+1) for e in t["A"]) + "}" if t["A"] else "∅"
            B   = "{" + ",".join(str(e+1) for e in t["B"]) + "}" if t["B"] else "∅"
            img = "{" + ",".join(str(e+1) for e in t["image"]) + "}" if t["image"] else "∅"
            print(f"    φ({A}) = {A} ∪ {B} = {img}")
        print(f"  LLM reason: {ex['reason']}")

    # ── 4. Failure cases — hard families to add as test cases ─────────────────
    failures = [r for r in records if not r["injective"]]
    print(f"\n── Failure cases ({len(failures)}) — candidate hard test families ─")
    seen = set()
    for r in failures[:5]:
        key = str(r["family"])
        if key in seen:
            continue
        seen.add(key)
        sets_str = ", ".join(
            "{" + ",".join(str(e+1) for e in sorted(F)) + "}" if F else "∅"
            for F in r["family"]
        )
        print(f"  coverage={r['coverage']:.0%}  rule={r['rule']}  "
              f"element=x{r['element']}  family=[{sets_str}]")

    print(f"\n→ To add new rules: inspect mapping_trace in novel_success records,")
    print(f"  find recurring B-selection patterns, name them, add to RULES in rl_env.py.")


if __name__ == "__main__":
    main()
