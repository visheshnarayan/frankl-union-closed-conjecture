"""
collect_traces.py — Sweep the LLM agent over (n, episodes) configs.

Runs llm_agent across a grid of (ground-set size, episode count) pairs,
writing one JSONL trace file per config to traces/.

Usage:
    python3 collect_traces.py                # run default sweep
    python3 collect_traces.py --dry-run      # print plan, exit
    python3 collect_traces.py --quiet        # suppress per-episode lines
"""

import argparse
import json
import os
from datetime import datetime

from rl_env import FranklEnv
from llm_agent import run_episode, build_inference_fn, DEFAULT_HF_MODEL

# ── Sweep config ──────────────────────────────────────────────────────────────
# (n, n_episodes)
# n=2/3  → tiny families, quick; good for seeding the dataset
# n=4    → main target; we can enumerate all 4,542 families eventually
# n=5    → brute force is infeasible (~6h), so random sampling is the only option
SWEEP = [
    (2,   50),
    (3,  150),
    (4,  500),
    (5,  200),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print plan without running any episodes")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress per-episode status lines")
    parser.add_argument("--backend",     choices=["ollama", "hf"], default="ollama",
                        help="ollama (local server) or hf (HuggingFace direct, GPU cluster)")
    parser.add_argument("--hf-model",    default=DEFAULT_HF_MODEL,
                        help="HuggingFace model ID (--backend hf only)")
    args = parser.parse_args()

    total_ep = sum(ep for _, ep in SWEEP)
    print(f"Sweep plan — {len(SWEEP)} configs, {total_ep} total episodes")
    print(f"{'n':>4}  {'episodes':>10}")
    print("-" * 20)
    for n, ep in SWEEP:
        print(f"  n={n}  ep={ep:>6}")
    print()

    if args.dry_run:
        print("(dry-run — exiting)")
        return

    inference_fn, model_name = build_inference_fn(args.backend, args.hf_model)
    print(f"Backend: {args.backend}  |  Model: {model_name}\n")

    os.makedirs("traces", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for n, n_episodes in SWEEP:
        trace_path = f"traces/sweep_{ts}_n{n}_ep{n_episodes}.jsonl"
        print(f"── n={n}, {n_episodes} episodes → {trace_path}")

        env    = FranklEnv(n=n, max_steps=1, seed=42)
        proved = 0

        skipped = 0
        with open(trace_path, "w") as tf:
            for ep in range(1, n_episodes + 1):
                try:
                    r = run_episode(env, ep,
                                    inference_fn=inference_fn,
                                    model_name=model_name,
                                    show_think=False,
                                    temperature=args.temperature)
                except Exception as e:
                    print(f"\n  [ep {ep} error: {e}] skipping")
                    skipped += 1
                    continue

                record = {k: v for k, v in r.items() if k != "think"}
                tf.write(json.dumps(record) + "\n")
                tf.flush()

                if r["injective"]:
                    proved += 1

                if not args.quiet:
                    status = "✓" if r["injective"] else "✗"
                    bar    = int(30 * ep / n_episodes)
                    print(f"  [{'█'*bar}{'░'*(30-bar)}] {ep}/{n_episodes}  "
                          f"{status}  rule={r['rule']:<22}  "
                          f"proved={proved}  skipped={skipped}",
                          end="\r", flush=True)

        ran = n_episodes - skipped
        print(f"\n  n={n}: {proved}/{ran} proved ({100*proved/ran:.1f}% of {ran} ran, {skipped} skipped)\n")

    print(f"All traces written to traces/")
    print(f"Next step: python3 build_dataset.py")


if __name__ == "__main__":
    main()
