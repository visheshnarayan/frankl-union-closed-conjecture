"""
Enumerate all nontrivial union-closed families on [n] and verify
Frankl's union-closed sets conjecture for each one.

A family F ⊆ P([n]) is:
  - union-closed  : A,B ∈ F  =>  A∪B ∈ F
  - nontrivial    : ∪F = [n]  (every element appears somewhere)

The conjecture: some element x appears in >= |F|/2 sets.
"""

from itertools import combinations


def subsets(n):
    """All subsets of [n] as frozensets, including empty set."""
    ground = list(range(1, n + 1))
    result = []
    for r in range(len(ground) + 1):
        for c in combinations(ground, r):
            result.append(frozenset(c))
    return result


def is_union_closed(family):
    fset = set(family)
    for A in family:
        for B in family:
            if A | B not in fset:
                return False
    return True


def is_nontrivial(family, n):
    ground = frozenset(range(1, n + 1))
    return frozenset().union(*family) == ground


def element_frequencies(family, n):
    """Return dict: element -> number of sets in family containing it."""
    return {x: sum(1 for F in family if x in F) for x in range(1, n + 1)}


def conjecture_holds(family, n):
    """True if some element appears in >= |F|/2 sets."""
    threshold = len(family) / 2
    freqs = element_frequencies(family, n)
    return any(v >= threshold for v in freqs.values())


def best_element(family, n):
    """Return (element, frequency, fraction) for the most frequent element."""
    freqs = element_frequencies(family, n)
    x = max(freqs, key=freqs.get)
    return x, freqs[x], freqs[x] / len(family)


def enumerate_families(n):
    all_sets = subsets(n)
    total_sets = len(all_sets)

    families_checked = 0
    conjecture_verified = 0
    close_calls = []          # families where best element is in exactly 1/2
    singleton_cases = 0       # families containing a singleton

    print(f"Ground set: [{n}]  |  Total subsets: {total_sets}")
    print("-" * 60)

    # Iterate over all non-empty sub-collections of P([n])
    # For n<=4 this is 2^16 = 65536 which is fast; n=5 is 2^32, too slow.
    # We always include the full set [n] (required by nontriviality + union-closed).
    ground = frozenset(range(1, n + 1))
    remaining = [s for s in all_sets if s != ground]

    for r in range(0, len(remaining) + 1):
        for combo in combinations(remaining, r):
            family = list(combo) + [ground]

            if not is_union_closed(family):
                continue
            if not is_nontrivial(family, n):
                continue

            families_checked += 1
            holds = conjecture_holds(family, n)
            if holds:
                conjecture_verified += 1

            x, freq, frac = best_element(family, n)
            if abs(frac - 0.5) < 1e-9:
                close_calls.append((family, x, freq, frac))

            has_singleton = any(len(F) == 1 for F in family)
            if has_singleton:
                singleton_cases += 1

    print(f"Nontrivial union-closed families found : {families_checked}")
    print(f"Conjecture verified                    : {conjecture_verified}")
    print(f"Conjecture holds for all               : {families_checked == conjecture_verified}")
    print(f"Families with a singleton set          : {singleton_cases}")
    print(f"Families where best element hits = 1/2 : {len(close_calls)}")

    if close_calls:
        print("\nClose-call examples (best element at exactly 1/2):")
        for fam, x, freq, frac in close_calls[:5]:
            sets_str = ", ".join("{" + ",".join(str(e) for e in sorted(F)) + "}"
                                 for F in sorted(fam, key=lambda s: (len(s), sorted(s))))
            print(f"  F = [{sets_str}]")
            print(f"      best element={x}, freq={freq}/{len(fam)}")

    return families_checked, conjecture_verified


if __name__ == "__main__":
    for n in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"  n = {n}")
        print(f"{'='*60}")
        enumerate_families(n)
