# Reading List — Frankl's Union-Closed Sets Conjecture
**Math 475 · Due Nov 12**

Work through these in order. Each item has a goal — you don't need to read everything
exhaustively, just get what's listed under "focus on".

---

## Day 1 — Intuition (Nov 8)

### 1. Gowers Blog Post
**File:** `~/Downloads/Frankl's union-closed conjecture — a possible Polymath project_.pdf`
**Pages:** All (~39 pages, but skim after page 10)
**Time:** ~45 min

Focus on:
- [ ] Why is the problem hard despite its simple statement?
- [ ] Why does the naive averaging argument fail? (pages 1–2, he gives a concrete example)
- [ ] What is a "weighted average" approach and why does it reduce to something trivial?
- [ ] The entropy-based distribution idea (page 3) — don't worry about details, just the concept
- [ ] Skim the comments section for other failed approaches mentioned

Skip: the dense comment thread mathematics deep dives — those are research-level

---

### 2. Your own notebook output
**File:** `~/Documents/math475-frankl/explore.ipynb`
**Time:** ~20 min

Focus on:
- [ ] Look at every tight case printed in Section 7
- [ ] Ask yourself: *why* are they all `{∅, A, [n]\A, [n]}` or the full power set?
- [ ] Try to sketch a proof of why these are the only tight families for n=3

---

## Day 2 — Known Results (Nov 9)

### 3. Nagel (2023) — Introduction and Frequencies
**File:** `~/Downloads/union-closed-set-conj.pdf`
**Pages:** 1–5 (§1 Introduction, §2 Frequencies)
**Time:** ~1 hour

Focus on:
- [ ] §1: Conjecture 1.1 — make sure you can state it cold without looking
- [ ] §1: Why can we assume `∅ ∈ F` WLOG? (paragraph after Conjecture 1.1)
- [ ] §2: The ordering of elements by frequency (eq. 2.1) — understand what this buys us
- [ ] §2: **Lemma 2.4** — this is the key lemma. Read the proof carefully.
  - The map φ: X ↦ X ∪ A is the core idea
  - Understand why the fiber φ⁻¹(Y) has size exactly 2^{#A−1}
- [ ] §2: **Theorem 2.6** (Question 2.1 holds for k=n and k=n−1) — read the proof
- [ ] §2: Question 2.1 — understand what it's asking and why k=1 recovers the conjecture

Skip: the full proof of Question 2.1 by induction (still open)

---

### 4. Nagel (2023) — Equivalent Structures
**File:** `~/Downloads/union-closed-set-conj.pdf`
**Pages:** 5–7 (§3.1 Equivalent Structures)
**Time:** ~45 min

Focus on:
- [ ] The intersection-closed dual — why is this equivalent? (one paragraph, easy)
- [ ] The interior operator τ(X) = ∪{F ∈ F : F ⊆ X} — verify the three properties
  (exclusivity, monotonicity, idempotence) on a small example
- [ ] **Theorem 3.1** — the bijection between UC families and interior operators
  (you don't need to read the proof, just understand the statement)
- [ ] Congruence classes T(F) = τ⁻¹(F) — what does it mean for two sets to be in the same class?

Skip: Theorem 3.2 and Corollary 3.3 (congruence relations) unless you have extra time

---

## Day 3 — Go Deep on One Proof (Nov 10)

### 5. Nagel (2023) — The Up-Set Weakening
**File:** `~/Downloads/union-closed-set-conj.pdf`
**Pages:** 9–12 (§3.2 Up-sets, §3.3 Demonstration, Theorem 3.7 and 3.10)
**Time:** ~1.5 hours — this is the hardest read, go slow

Focus on:
- [ ] Definition of an **up-set** (§3.2, p.10) — simple, understand it cold
- [ ] Why is `[x, [n]] = {X ⊆ [n] : x ∈ X}` an up-set? (and what does it have to do with FUNC?)
- [ ] **Theorem 3.7** statement — this is a weakening: *why* is it weaker than FUNC?
- [ ] **Corollary 3.6** — the ordering on F by congruence class size (needed for Thm 3.10 proof)
- [ ] **Lemma 3.9** — the floor-function inequality. Read the proof once; it's elementary calculus.
- [ ] **Theorem 3.10** proof — this is the main result. Work through it step by step.
  - U is defined as the union of T(F_i) for the first ⌈m/t⌉ sets
  - Two things to verify: (a) U is an up-set, (b) #U ≤ ⌈2^n/t⌉
  - Setting t=2 gives Theorem 3.7

Write down: in your own words, what is the proof strategy? Why does it work?

---

### 6. Bruhn–Schaudt Survey (skim)
**URL:** referenced as [7] in Nagel — `http://www.zaik.uni-koeln.de/~schaudt/UCSurvey.pdf`
**Pages:** 1–8 only (Introduction + §2 Basic observations)
**Time:** ~30 min

Focus on:
- [ ] The timeline: when was the conjecture posed, who made progress when?
- [ ] §2: "Basic observations" — especially the small cases (n=2, singleton, two-element)
- [ ] Any reformulations not in Nagel

Skip: everything after §3 (too detailed for the report)

---

## Day 4 — Write (Nov 11)

No new reading. Use what you learned to write:

- [ ] **§3 Approaches and Ideas Explored** (~2–3 pages)
  - Gowers's averaging failure → weighted average → why it's hard
  - Nagel's up-set weakening: explain the idea and state Theorem 3.7
  - Walk through Lemma 2.4 (the fiber argument) — this is a good self-contained proof to present

- [ ] **§4 Progress and Obstacles** (~1–2 pages)
  - Your computation: what you found, the tight cases, the structural observation
  - Why n=5 is infeasible by brute force
  - What you tried to prove and where you got stuck

- [ ] Proofread, check all citations, verify page count ≤ 10

---

## Reference at a Glance

| Source | What it's for |
|--------|--------------|
| Gowers blog | Intuition, failed approaches, "why is this hard" |
| Nagel §1–2 | Formal setup, Lemma 2.4, frequency bounds |
| Nagel §3.1 | Equivalent structures (interior operators) |
| Nagel §3.2–3.3 | Up-set weakening — the main proof to understand |
| Bruhn–Schaudt | History, timeline, broader survey |
| Your notebook | Original computation — cite this as your own work |

---

## Key Facts to Know Cold (for the report and presentation)

- The conjecture: some element in ≥ 1/2 of sets
- Equivalent: intersection-closed dual (element in ≤ 1/2 of sets)
- Singleton case: one-line injection proof
- Verified computationally up to |F| ≤ 50 (Bruhn–Schaudt)
- Best known bound on max frequency: Ω(√(log n / n)) · m
- Tight cases: `{∅, A, [n]\A, [n]}` and `P([n])` — balance is extremal
- Open since ~1979, no proof or counterexample known
