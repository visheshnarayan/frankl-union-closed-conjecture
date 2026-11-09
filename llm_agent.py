"""
LLM Agent for FranklEnv using DeepSeek-R1 (via Ollama).

The model receives a natural-language description of the current
union-closed family, its Φ-scores, and the available mapping rules.
It reasons (chain-of-thought) and returns a (element, rule) action.

Usage:
    python3 llm_agent.py                    # run 10 episodes, show reasoning
    python3 llm_agent.py --episodes 50      # benchmark
    python3 llm_agent.py --quiet            # suppress <think> blocks
"""

import argparse
import json
import os
import re
import requests
import time
from collections import defaultdict
from datetime import datetime

from rl_env import FranklEnv, RULES, N_RULES, phi_scores, all_subsets

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "llama3.2:3b"


# ── Prompt builder ────────────────────────────────────────────────────────────

def family_to_str(family, n):
    """Human-readable list of sets."""
    lines = []
    for F in sorted(family, key=lambda s: (len(s), sorted(s))):
        if not F:
            lines.append("  ∅")
        else:
            lines.append("  {" + ", ".join(str(e + 1) for e in sorted(F)) + "}")
    return "\n".join(lines)

def build_prompt(family, n, phi, freqs, episode: int) -> str:
    phi_str  = " | ".join(f"x{i+1}={v:.2f}" for i, v in enumerate(phi))
    freq_str = " | ".join(f"x{i+1}:{freqs[i]}/{len(family)}" for i in range(n))
    sets_str = ", ".join(
        "{" + ",".join(str(e+1) for e in sorted(F)) + "}" if F else "∅"
        for F in sorted(family, key=lambda s: (len(s), sorted(s)))
    )
    rules_str = " | ".join(f"{i}:{r}" for i, r in enumerate(RULES))

    return f"""Frankl conjecture proof search (ep {episode}).

Family F on [1..{n}]: {sets_str}
Frequencies: {freq_str}
Phi-scores (bijection signal): {phi_str}

Goal: pick element x and rule r to build injective φ(A)=A∪B(A) mapping F_bad→F_good.
Rules: {rules_str}
Avoid rules 0 (union_min) and 6 (singleton_injection) — known strategies.

Reply with ONLY valid JSON, no markdown:
{{"element":<1-{n}>,"rule":<0-{N_RULES-1}>,"reason":"<one sentence>"}}"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, temperature: float = 0.3) -> tuple[str, str]:
    """
    Returns (think_block, final_answer).
    DeepSeek-R1 wraps chain-of-thought in <think>...</think>.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    # Split <think> block from final answer
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return think, answer


def parse_action(answer: str, n: int) -> tuple[int, int, str] | None:
    """Parse JSON response → (x_0indexed, rule_idx, reason). None on failure."""
    # strip markdown fences if model added them anyway
    answer = re.sub(r"```[a-z]*", "", answer).strip().strip("`")
    try:
        obj     = json.loads(answer)
        x       = int(obj["element"]) - 1          # convert to 0-indexed
        rule    = int(obj["rule"])
        reason  = obj.get("reason", "")
        if 0 <= x < n and 0 <= rule < N_RULES:
            return x, rule, reason
    except Exception:
        pass
    return None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: FranklEnv, episode: int, show_think: bool = True,
                temperature: float = 0.3) -> dict:
    obs, _ = env.reset()
    n      = env.n
    family = env.family

    phi   = phi_scores(family, n)
    freqs = [sum(1 for F in family if x in F) for x in range(n)]

    prompt = build_prompt(family, n, phi, freqs, episode)
    t0     = time.perf_counter()
    think, answer = call_ollama(prompt, temperature)
    elapsed = time.perf_counter() - t0

    parsed = parse_action(answer, n)
    if parsed is None:
        # Fallback: best Φ-score element, union_closest rule
        x_fb = int(phi.argmax())
        action = x_fb * N_RULES + 2
        print(f"  [parse fail] fallback action: x={x_fb+1}, union_closest")
        result = {"episode": episode, "parse_fail": True,
                  "think": think, "raw": answer, "elapsed": elapsed}
        obs, reward, _, _, info = env.step(action)
        result.update({"reward": reward, "injective": info["injective"],
                        "rule": info["rule"], "element": x_fb + 1,
                        "reason": "fallback", "coverage": info["coverage"]})
        return result

    x_idx, rule_idx, reason = parsed
    action = x_idx * N_RULES + rule_idx
    obs, reward, _, _, info = env.step(action)

    return {
        "episode":   episode,
        "element":   x_idx + 1,
        "rule":      RULES[rule_idx],
        "reason":    reason,
        "reward":    reward,
        "injective": info["injective"],
        "known":     info.get("known", False),
        "coverage":  info["coverage"],
        "elapsed":   elapsed,
        "think":     think,
        "raw":       answer,
        "parse_fail": False,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,   default=10)
    parser.add_argument("--n",           type=int,   default=4)
    parser.add_argument("--quiet",       action="store_true",
                        help="suppress <think> blocks")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    env = FranklEnv(n=args.n, max_steps=1, seed=99)

    results      = []
    rule_counts  = defaultdict(int)
    rule_success = defaultdict(int)

    # ── Log file setup ────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{MODEL.replace(':', '-')}_{ts}_n{args.n}_ep{args.episodes}.txt"
    log_file = open(log_path, "w")

    def log(line=""):
        print(line)
        log_file.write(line + "\n")

    log(f"Run started : {datetime.now().isoformat()}")
    log(f"Model       : {MODEL}")
    log(f"n           : {args.n}")
    log(f"Episodes    : {args.episodes}")
    log("=" * 70)

    for ep in range(1, args.episodes + 1):
        r = run_episode(env, ep, show_think=not args.quiet,
                        temperature=args.temperature)
        results.append(r)
        rule_counts[r["rule"]] += 1
        if r["injective"]:
            rule_success[r["rule"]] += 1

        status = "✓ PROVED" if r["injective"] else "✗ failed"
        known  = " [KNOWN]" if r.get("known") else ""
        pf     = " [parse fail]" if r["parse_fail"] else ""
        log(f"\nEp {ep:>3}  {status}{known}{pf}")
        log(f"  Element : x{r['element']}")
        log(f"  Rule    : {r['rule']}")
        log(f"  Reason  : {r['reason']}")
        log(f"  Reward  : {r['reward']:.3f}   Coverage: {r['coverage']:.1%}   "
            f"Time: {r['elapsed']:.1f}s")

        if not args.quiet and r["think"]:
            think_preview = r["think"][:600].replace("\n", " ")
            log(f"  <think> : {think_preview}…")

        # Full think block to log file only
        if r["think"]:
            log_file.write(f"  <think_full>\n{r['think']}\n  </think_full>\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_proved     = sum(1 for r in results if r["injective"])
    n_known      = sum(1 for r in results if r.get("known"))
    mean_reward  = sum(r["reward"] for r in results) / len(results)
    mean_elapsed = sum(r["elapsed"] for r in results) / len(results)

    log("\n" + "=" * 70)
    log(f"SUMMARY  ({args.episodes} episodes, n={args.n})")
    log(f"  Proved          : {n_proved}/{args.episodes}  ({100*n_proved/args.episodes:.1f}%)")
    log(f"  Used known rule : {n_known}/{args.episodes}")
    log(f"  Mean reward     : {mean_reward:.4f}")
    log(f"  Mean LLM time   : {mean_elapsed:.1f}s / episode")

    log(f"\n  {'Rule':<25} {'Chosen':>7} {'Succeeded':>10} {'Rate':>7}")
    log("  " + "-" * 53)
    for rule in RULES:
        c = rule_counts[rule]
        s = rule_success[rule]
        if c == 0:
            continue
        flag = " ← known" if rule in ("union_min", "singleton_injection") else ""
        log(f"  {rule:<25} {c:>7} {s:>10} {s/c:>7.1%}{flag}")

    log(f"\nLog saved to: {log_path}")
    log_file.close()

    # Also save full results as JSON
    json_path = log_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({"model": MODEL, "n": args.n, "episodes": args.episodes,
                   "results": [{k: v for k, v in r.items() if k != "think"}
                                for r in results]}, f, indent=2)


if __name__ == "__main__":
    main()
