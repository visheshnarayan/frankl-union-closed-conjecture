"""
Timed enumeration of nontrivial union-closed families on [n].

Runs n=2,3,4 fully. For n>=5, samples a random slice of the search
space to estimate the full runtime, then extrapolates.

Usage:
    python3 timing.py
"""

import time
import math
import random
from itertools import combinations


# ── core combinatorics (same as enumerate.py) ────────────────────────────────

def subsets(n):
    ground = list(range(1, n + 1))
    return [frozenset(c) for r in range(len(ground) + 1)
            for c in combinations(ground, r)]

def is_union_closed(family):
    s = set(family)
    return all(A | B in s for A in family for B in family)

def is_nontrivial(family, n):
    return frozenset().union(*family) == frozenset(range(1, n + 1))

def element_freqs(family, n):
    return {x: sum(1 for F in family if x in F) for x in range(1, n + 1)}

def conjecture_holds(family, n):
    m = len(family)
    freqs = element_freqs(family, n)
    return any(v >= m / 2 for v in freqs.values())


# ── full enumeration (n <= 4) ─────────────────────────────────────────────────

def run_full(n):
    ground = frozenset(range(1, n + 1))
    all_sets = subsets(n)
    remaining = [s for s in all_sets if s != ground]
    candidates = 2 ** len(remaining)

    families = 0
    violations = 0

    t0 = time.perf_counter()
    for r in range(len(remaining) + 1):
        for combo in combinations(remaining, r):
            family = list(combo) + [ground]
            if not is_union_closed(family):
                continue
            if not is_nontrivial(family, n):
                continue
            families += 1
            if not conjecture_holds(family, n):
                violations += 1
    elapsed = time.perf_counter() - t0

    return {
        'n': n,
        'candidates': candidates,
        'families': families,
        'violations': violations,
        'elapsed': elapsed,
        'estimated': False,
    }


# ── sampled estimation (n >= 5) ───────────────────────────────────────────────

def run_sampled(n, sample_size=5_000, seed=42):
    """
    Sample `sample_size` random candidate families and time them.
    Extrapolate to estimate the full runtime.
    """
    random.seed(seed)
    ground = frozenset(range(1, n + 1))
    all_sets = subsets(n)
    remaining = [s for s in all_sets if s != ground]
    total_candidates = 2 ** len(remaining)

    t0 = time.perf_counter()
    checked = 0
    families_in_sample = 0

    for _ in range(sample_size):
        # pick a random subset of `remaining` to form a candidate family
        r = random.randint(0, len(remaining))
        combo = random.sample(remaining, r)
        family = combo + [ground]
        checked += 1
        if is_union_closed(family) and is_nontrivial(family, n):
            families_in_sample += 1

    elapsed_sample = time.perf_counter() - t0
    time_per_candidate = elapsed_sample / sample_size
    estimated_total = time_per_candidate * total_candidates
    estimated_families = int(families_in_sample / sample_size * total_candidates)

    return {
        'n': n,
        'candidates': total_candidates,
        'families': estimated_families,
        'violations': '?',
        'elapsed': estimated_total,
        'estimated': True,
    }


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} hrs"
    if seconds < 86400 * 365:
        return f"{seconds/86400:.1f} days"
    if seconds < 86400 * 365 * 1000:
        return f"{seconds/86400/365:.1f} yrs"
    return f"{seconds/86400/365:.2e} yrs"

def fmt_int(n):
    if isinstance(n, str):
        return n
    if n >= 1e12:
        return f"{n:.2e}"
    return f"{n:,}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = []

    print("Running full enumeration for n = 2, 3, 4 ...")
    for n in [2, 3, 4]:
        r = run_full(n)
        results.append(r)
        status = "✓ no violations" if r['violations'] == 0 else f"✗ {r['violations']} violations"
        print(f"  n={n}: {r['families']:,} families in {fmt_time(r['elapsed'])}  [{status}]")

    print("\nEstimating runtime for n = 5, 6, 7 via sampling ...")
    for n in [5, 6, 7]:
        r = run_sampled(n, sample_size=2_000)
        results.append(r)
        print(f"  n={n}: ~{fmt_int(r['families'])} families (est.), "
              f"full run would take ~{fmt_time(r['elapsed'])}")

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"{'n':>3}  {'Candidates':>18}  {'UC Families':>14}  "
          f"{'Runtime':>12}  {'Conjecture':>12}  {'Method'}")
    print("-" * 78)
    for r in results:
        conj = ("✓ verified" if r['violations'] == 0
                else ("? unknown" if r['violations'] == '?' else f"✗ {r['violations']} fail"))
        method = "estimated" if r['estimated'] else "exact"
        print(f"{r['n']:>3}  {fmt_int(r['candidates']):>18}  "
              f"{fmt_int(r['families']):>14}  "
              f"{fmt_time(r['elapsed']):>12}  {conj:>12}  {method}")
    print("=" * 78)

    # ── candidate count projection ────────────────────────────────────────────
    print()
    print("Candidate family counts  (2^(2^n - 1)):")
    for n in range(2, 11):
        exp = 2**n - 1
        count = 2**exp if exp <= 60 else float('inf')
        feasible = "✓ feasible" if n <= 4 else ("~feasible" if n == 5 else "✗ infeasible")
        count_str = f"2^{exp}" if exp > 20 else fmt_int(2**exp)
        print(f"  n={n:>2}: {count_str:>25}  {feasible}")


if __name__ == "__main__":
    main()
