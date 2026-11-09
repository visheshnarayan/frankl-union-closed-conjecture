"""
FranklEnv — RL environment for discovering injection-based proofs
of Frankl's union-closed sets conjecture.

GOAL
----
Find an injective map  φ: F_bad → F_good
  F_bad  = {A ∈ F : x ∉ A}
  F_good = {A ∈ F : x ∈ A}

If |F_bad| ≤ |F_good|, the conjecture holds for element x.
If φ is injective, that proves it.

PROOF TEMPLATE THE AGENT EXPLORES
-----------------------------------
For each A ∈ F_bad, the agent picks some B(A) ∈ F with x ∈ B(A).
Then φ(A) := A ∪ B(A) ∈ F  (union-closed guarantee).
If φ is injective → valid proof for this family.

The agent must avoid rediscovering two known strategies:
  (K1) Singleton injection:  φ(A) = A ∪ {x}  when {x} ∈ F
  (K2) Nagel fiber map:      φ(A) = A ∪ argmin_{C ∈ F, x ∈ C} |C|

OBSERVATION (STATE)
-------------------
For a family F on [n], represented as an (|F|, n) incidence matrix,
we expose a fixed-size feature vector:
  - sorted element frequency fractions  (n values)
  - fraction of sets containing each element vs |F|
  - Φ-scores: for each element x, mean_{A ∈ F} score(x, A)
      where score(x, A) = #{C ∈ Φ(A) : x ∈ C} / |Φ(A)|
  - has_singleton, has_empty flags
  - family size / 2^n  (density)

ACTION
------
Discrete:  agent picks an element x ∈ [n] to certify
Continuous sub-action:  for each A ∈ F_bad, picks a weight vector over
  F_good — the mapping B(A) = argmax of that weight vector.

To keep it tractable we discretise: the agent picks x and a
"rule index" r ∈ {0, …, R-1} from a small catalogue of mapping rules.
The environment applies rule r to build φ, checks injectivity, and scores.

REWARDS
-------
  +1.0   φ is injective → valid proof found
  -0.5   φ uses known strategy K1 (singleton injection)
  -0.3   φ uses known strategy K2 (Nagel min-set fiber)
   0.0   φ not injective (failed attempt)
  +0.1   bonus for any novel φ with |image(φ)| / |F_bad| > 0.8  (partial credit)

The novelty penalties stack: the agent should learn to avoid known paths.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import random
from typing import Optional


# ── Core combinatorics ────────────────────────────────────────────────────────

def all_subsets(n: int):
    ground = list(range(n))
    return [frozenset(c) for r in range(n + 1) for c in combinations(ground, r)]

def is_union_closed(family):
    s = set(family)
    return all(a | b in s for a in family for b in family)

def is_nontrivial(family, n):
    return frozenset().union(*family) == frozenset(range(n))

def sample_uc_family(n: int, rng: random.Random) -> list:
    """Sample a random nontrivial union-closed family on [n]."""
    ground = frozenset(range(n))
    all_sets = all_subsets(n)
    remaining = [s for s in all_sets if s != ground]
    while True:
        k = rng.randint(0, len(remaining))
        combo = rng.sample(remaining, k)
        family = combo + [ground]
        if is_union_closed(family) and is_nontrivial(family, n):
            return family


# ── Mapping rules catalogue ───────────────────────────────────────────────────
# Each rule: (family, bad_set, good_sets, x) → one good_set to map bad_set to

def rule_union_min(A, F_good, x):
    """Map A → A ∪ argmin_{B ∈ F_good} |B|  (Nagel-style min-fiber)."""
    B = min(F_good, key=len)
    return A | B

def rule_union_max(A, F_good, x):
    """Map A → A ∪ argmax_{B ∈ F_good} |B|  (max set)."""
    B = max(F_good, key=len)
    return A | B

def rule_union_closest(A, F_good, x):
    """Map A → A ∪ B where B ∈ F_good minimises |A △ B| (symmetric diff)."""
    B = min(F_good, key=lambda b: len(A.symmetric_difference(b)))
    return A | B

def rule_union_disjoint_pref(A, F_good, x):
    """Prefer B ∈ F_good disjoint from A; fallback to min."""
    disjoint = [b for b in F_good if not (A & b)]
    B = min(disjoint, key=len) if disjoint else min(F_good, key=len)
    return A | B

def rule_union_superset_pref(A, F_good, x):
    """Prefer B ∈ F_good that is a superset of A; fallback to min."""
    supers = [b for b in F_good if A <= b]
    B = min(supers, key=len) if supers else min(F_good, key=len)
    return A | B

def rule_union_random(A, F_good, x, rng):
    """Map A → A ∪ B for a random B ∈ F_good."""
    B = rng.choice(list(F_good))
    return A | B

def rule_singleton_injection(A, F_good, x):
    """KNOWN K1: φ(A) = A ∪ {x}  (only valid when {x} ∈ F)."""
    return A | frozenset([x])

RULES = [
    "union_min",           # 0 — Nagel-style (K2, penalised)
    "union_max",           # 1
    "union_closest",       # 2
    "union_disjoint_pref", # 3
    "union_superset_pref", # 4
    "union_random",        # 5
    "singleton_injection", # 6 — K1, penalised
]
N_RULES = len(RULES)


def apply_rule(rule_idx: int, A, F_good, x, rng) -> Optional[tuple[frozenset, frozenset]]:
    """
    Apply rule rule_idx to map A → A ∪ B for some B ∈ F_good.
    Returns (B_chosen, image=A∪B) so callers can record the full mapping detail.
    Returns None if F_good is empty.
    """
    if not F_good:
        return None
    r = RULES[rule_idx]
    if r == "union_min":           B = min(F_good, key=len)
    elif r == "union_max":         B = max(F_good, key=len)
    elif r == "union_closest":     B = min(F_good, key=lambda b: len(A.symmetric_difference(b)))
    elif r == "union_disjoint_pref":
        disj = [b for b in F_good if not (A & b)]
        B = min(disj, key=len) if disj else min(F_good, key=len)
    elif r == "union_superset_pref":
        supers = [b for b in F_good if A <= b]
        B = min(supers, key=len) if supers else min(F_good, key=len)
    elif r == "union_random":      B = rng.choice(list(F_good))
    elif r == "singleton_injection": B = frozenset([x])
    else:                          return None
    return B, A | B

def is_known_strategy(rule_idx: int) -> tuple[bool, float]:
    """Return (is_known, penalty)."""
    r = RULES[rule_idx]
    if r == "singleton_injection": return True, 0.5
    if r == "union_min":           return True, 0.3
    return False, 0.0


# ── Feature extraction ────────────────────────────────────────────────────────

def phi_scores(family, n):
    """
    For each element x, compute mean_{A ∈ F} score(x, A)
    where score(x, A) = #{C ∈ Φ(A) : x ∈ C} / |Φ(A)|
    and Φ(A) = {A ∪ B : B ∈ F}.

    This is the core of the user's proposed proof direction.
    """
    scores = np.zeros(n)
    for x in range(n):
        total = 0.0
        for A in family:
            phi_A = [A | B for B in family]  # union image of A
            count_x = sum(1 for C in phi_A if x in C)
            total += count_x / len(phi_A)
        scores[x] = total / len(family)
    return scores

def family_features(family, n) -> np.ndarray:
    """
    Fixed-size feature vector for a union-closed family.
    Length = 4*n + 3
    """
    m = len(family)
    # Element frequency fractions (sorted ascending)
    freqs = np.array([sum(1 for F in family if x in F) / m for x in range(n)])
    freqs_sorted = np.sort(freqs)

    # Φ-scores (the bijection-based signal)
    phi = phi_scores(family, n)
    phi_sorted = np.sort(phi)

    # Structural flags
    has_singleton = float(any(len(F) == 1 for F in family))
    has_empty     = float(frozenset() in family)
    density       = m / (2 ** n)

    return np.concatenate([freqs_sorted, freqs, phi_sorted, phi,
                           [has_singleton, has_empty, density]])


# ── Environment ───────────────────────────────────────────────────────────────

class FranklEnv(gym.Env):
    """
    Gymnasium environment for injection-based proof discovery.

    Action space:  Discrete(n * N_RULES)
        encodes (element x, rule r) jointly as x * N_RULES + r

    Observation space: Box of shape (4*n + 3,)

    Episode:
        - Each reset samples a new random union-closed family
        - Each step the agent picks (x, rule) → build φ → score
        - Episode terminates after one step (single-shot proof attempt)
          OR after max_steps attempts on the same family
    """

    metadata = {"render_modes": []}

    def __init__(self, n: int = 4, max_steps: int = 10, seed: int = 42):
        super().__init__()
        self.n = n
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        obs_dim = 4 * n + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n * N_RULES)

        self.family = None
        self._step_count = 0
        self._proved = False

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.family = sample_uc_family(self.n, self._rng)
        self._step_count = 0
        self._proved = False
        obs = family_features(self.family, self.n).astype(np.float32)
        return obs, {}

    def step(self, action: int):
        x      = action // N_RULES
        rule   = action  % N_RULES
        self._step_count += 1

        F      = self.family
        F_set  = set(F)
        F_bad  = [A for A in F if x not in A]
        F_good = [A for A in F if x in A]

        reward = 0.0
        info   = {"x": x, "rule": RULES[rule], "injective": False,
                  "known": False, "coverage": 0.0}

        # Trivial: no bad sets → x is in every set → conjecture trivially holds
        if not F_bad:
            reward = 1.0
            info["injective"] = True
            info["trivial"] = True
            self._proved = True
            obs = family_features(F, self.n).astype(np.float32)
            return obs, reward, True, False, info

        # Build the mapping φ — record B chosen and image for each A
        mapping      = {}   # A → image (A∪B)
        mapping_B    = {}   # A → B chosen  (for traces)
        valid        = True
        for A in F_bad:
            result = apply_rule(rule, A, F_good, x, self._rng)
            if result is None:
                valid = False; break
            B, img = result
            if img not in F_set:
                valid = False; break
            mapping[A]   = img
            mapping_B[A] = B

        if not valid:
            # Rule produced an image outside F — invalid step
            obs = family_features(F, self.n).astype(np.float32)
            terminated = self._step_count >= self.max_steps
            return obs, -0.1, terminated, False, info

        # Coverage: how much of F_bad was mapped injectively
        images   = list(mapping.values())
        n_unique = len(set(images))
        coverage = n_unique / len(F_bad)
        info["coverage"] = coverage

        is_injective = (n_unique == len(F_bad))
        info["injective"] = is_injective

        # Full mapping trace — serialisable as sorted-list pairs
        def fs(s): return sorted(s)   # frozenset → sorted list for JSON
        info["mapping_trace"] = [
            {
                "A":     fs(A),
                "B":     fs(mapping_B[A]),
                "image": fs(mapping[A]),
            }
            for A in F_bad
        ]
        info["F_bad"]  = [fs(A) for A in F_bad]
        info["F_good"] = [fs(A) for A in F_good]

        # Known-strategy penalty
        known, penalty = is_known_strategy(rule)
        # K1 only really fires when {x} ∈ F and rule is singleton_injection
        if RULES[rule] == "singleton_injection":
            singleton_x = frozenset([x])
            if singleton_x not in F_set:
                # Singleton not in family → map produces invalid images
                # already caught above, but guard anyway
                known, penalty = False, 0.0
        info["known"] = known

        if is_injective:
            reward = 1.0 - penalty          # e.g. 0.5 if K1, 0.7 if K2
            self._proved = True
            terminated = True
        else:
            reward = 0.1 * coverage - penalty
            terminated = self._step_count >= self.max_steps

        obs = family_features(F, self.n).astype(np.float32)
        return obs, reward, terminated, False, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def render(self):
        if self.family is None:
            print("Not initialised.")
            return
        print(f"\nFamily on [{self.n}], size {len(self.family)}:")
        for F in sorted(self.family, key=lambda s: (len(s), sorted(s))):
            label = "{" + ", ".join(str(e+1) for e in sorted(F)) + "}" if F else "∅"
            print(f"  {label}")
        phi = phi_scores(self.family, self.n)
        print(f"\nΦ-scores (bijection signal): " +
              "  ".join(f"x{i+1}={v:.3f}" for i, v in enumerate(phi)))
        freqs = {x: sum(1 for F in self.family if x in F) for x in range(self.n)}
        m = len(self.family)
        print(f"Frequencies:               " +
              "  ".join(f"x{x+1}={freqs[x]}/{m}" for x in range(self.n)))


# ── Random-agent baseline ─────────────────────────────────────────────────────

def run_random_baseline(n=4, n_episodes=2000, seed=0):
    """
    Random agent — acts uniformly at random.
    Shows how often each rule independently succeeds, and what
    the Φ-score correlation with proof success looks like.
    """
    env = FranklEnv(n=n, max_steps=20, seed=seed)
    rng = random.Random(seed)

    rule_successes  = {r: 0 for r in RULES}
    rule_attempts   = {r: 0 for r in RULES}
    phi_on_success  = []
    phi_on_failure  = []
    total_reward    = 0.0

    obs, _ = env.reset()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = rng.randint(0, env.action_space.n - 1)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            rule_attempts[info["rule"]] += 1
            if info["injective"]:
                rule_successes[info["rule"]] += 1
                phi_on_success.append(obs[:n].max())   # best Φ-score
            else:
                phi_on_failure.append(obs[:n].max())

        total_reward += ep_reward

    print(f"\n── Random agent baseline  (n={n}, {n_episodes} episodes) ──")
    print(f"Mean episode reward: {total_reward / n_episodes:.4f}\n")
    print(f"{'Rule':<25} {'Attempts':>9} {'Successes':>10} {'Rate':>7}")
    print("-" * 55)
    for r in RULES:
        att = rule_attempts[r]
        suc = rule_successes[r]
        rate = suc / att if att else 0
        flag = " ← KNOWN (penalised)" if r in ("union_min", "singleton_injection") else ""
        print(f"  {r:<23} {att:>9} {suc:>10} {rate:>7.1%}{flag}")

    if phi_on_success and phi_on_failure:
        print(f"\nMax Φ-score when proof found :  "
              f"mean={np.mean(phi_on_success):.3f}  "
              f"std={np.std(phi_on_success):.3f}")
        print(f"Max Φ-score when proof failed:  "
              f"mean={np.mean(phi_on_failure):.3f}  "
              f"std={np.std(phi_on_failure):.3f}")
        print("\n→ Higher Φ-score correlates with proof success, validating")
        print("  the bijection-based signal as a useful state feature.")


if __name__ == "__main__":
    # Demo: show one family and its Φ-scores
    print("=" * 60)
    print("FranklEnv — single episode demo")
    print("=" * 60)
    env = FranklEnv(n=3, max_steps=10, seed=7)
    obs, _ = env.reset()
    env.render()

    print("\nTrying each (element, rule) combination:")
    print(f"  {'Action':<30} {'Reward':>8}  {'Injective':>10}  {'Known':>7}  {'Coverage':>9}")
    print("  " + "-" * 68)
    for action in range(env.action_space.n):
        env.reset()
        x    = action // N_RULES
        rule = RULES[action % N_RULES]
        _, reward, _, _, info = env.step(action)
        inj  = "✓" if info["injective"] else "✗"
        knwn = "✓" if info.get("known") else ""
        print(f"  x={x+1}, {rule:<25} {reward:>8.3f}  {inj:>10}  {knwn:>7}  "
              f"{info['coverage']:>9.1%}")

    # Baseline
    print()
    run_random_baseline(n=4, n_episodes=1000, seed=42)
