# Frankl's Union-Closed Sets Conjecture

**Math 475 Research Project · Fall 2026 · Vishesh Narayan**

---

## The Problem

A family of sets $\mathcal{F}$ is **union-closed** if $A, B \in \mathcal{F} \Rightarrow A \cup B \in \mathcal{F}$.

**Frankl's Conjecture (~1979):** Every finite, nontrivial union-closed family has an element that belongs to at least half of its sets.

Despite its elementary statement, this has been open for over 40 years. No proof or counterexample is known in general.

---

## Repo Structure

```
.
├── report.tex          # LaTeX source for the individual report (due Nov 12)
├── enumerate.py        # Brute-force enumerator — verifies conjecture for n ≤ 4
└── explore.ipynb       # Jupyter notebook with visualizations and analysis
```

---

## Running the Code

**Requirements:** Python 3, with a virtual environment set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter seaborn pandas matplotlib
```

**Enumerator (terminal output):**
```bash
python3 enumerate.py
```

**Notebook (interactive):**
```bash
jupyter notebook explore.ipynb
```

**Compile the report:**
```bash
pdflatex report.tex
```

---

## Computational Results

| n | Families found | Conjecture holds | Tight cases (best = 1/2) |
|---|----------------|-----------------|--------------------------|
| 2 | 8              | 8 / 8           | 2                        |
| 3 | 90             | 90 / 90         | 5                        |
| 4 | 4,542          | 4,542 / 4,542   | 15                       |

Tight cases are always complementary-pair families $\{\emptyset, A, [n]\setminus A, [n]\}$ or the full power set $\mathcal{P}([n])$ — matching the theoretical extremal construction from Nagel (2023).

---

## References

- H. Bruhn, O. Schaudt — *The Journey of the Union-Closed Sets Conjecture*, Graphs and Combinatorics 31 (2015)
- N. Nagel — *Notes on the Union Closed Sets Conjecture*, arXiv:2208.03803 (2023)
- W.T. Gowers — *Frankl's union-closed conjecture: a possible Polymath project?* (2016)
