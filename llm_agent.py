"""
LLM Agent for FranklEnv.

Supports two backends:
  ollama  (default) — calls a local Ollama server; good for local dev
  hf                — loads model directly via HuggingFace transformers;
                      works on GPU clusters with no Ollama required

Usage:
    python3 llm_agent.py                                  # Ollama, 10 eps
    python3 llm_agent.py --backend hf                     # HF inference
    python3 llm_agent.py --backend hf --hf-model Qwen/Qwen2.5-1.5B-Instruct
    python3 llm_agent.py --episodes 50 --quiet
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime

from rl_env import FranklEnv, RULES, N_RULES, phi_scores, all_subsets

OLLAMA_URL       = "http://localhost:11434/api/chat"
OLLAMA_MODEL     = "llama3.2:3b"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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


# ── Ollama backend ────────────────────────────────────────────────────────────

def call_ollama(prompt: str, temperature: float = 0.3) -> tuple[str, str]:
    """Call a local Ollama server. Returns (think_block, answer)."""
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"  [ollama timeout/conn: {type(e).__name__}] returning empty")
        return "", ""

    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think  = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return think, answer


# ── HuggingFace backend ────────────────────────────────────────────────────────

_hf_model     = None
_hf_tokenizer = None
_hf_model_id  = None


def load_hf_model(model_id: str):
    """Load model + tokenizer once; subsequent calls are no-ops."""
    global _hf_model, _hf_tokenizer, _hf_model_id
    if _hf_model is not None and _hf_model_id == model_id:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf] Loading {model_id}...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _hf_tokenizer.pad_token is None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    _hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    _hf_model.eval()
    _hf_model_id = model_id
    print(f"[hf] Model loaded on {next(_hf_model.parameters()).device}")


def call_hf(prompt: str, temperature: float = 0.3) -> tuple[str, str]:
    """Run inference via loaded HF model. Returns ("", answer)."""
    import torch
    inputs = _hf_tokenizer(prompt, return_tensors="pt").to(
        next(_hf_model.parameters()).device
    )
    with torch.no_grad():
        out = _hf_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=max(temperature, 1e-4),
            do_sample=(temperature > 0),
            pad_token_id=_hf_tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = _hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return "", answer   # no <think> block for standard chat models


# ── Backend factory ────────────────────────────────────────────────────────────

def build_inference_fn(backend: str, hf_model: str = DEFAULT_HF_MODEL):
    """Return (inference_fn, model_name_str)."""
    if backend == "hf":
        load_hf_model(hf_model)
        return call_hf, hf_model
    else:
        return call_ollama, OLLAMA_MODEL


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

def run_episode(env: FranklEnv, episode: int,
                inference_fn=None,
                model_name: str = OLLAMA_MODEL,
                show_think: bool = True,
                temperature: float = 0.3) -> dict:
    if inference_fn is None:
        inference_fn = call_ollama

    obs, _ = env.reset()
    n      = env.n
    family = env.family

    phi   = phi_scores(family, n)
    freqs = [sum(1 for F in family if x in F) for x in range(n)]

    prompt = build_prompt(family, n, phi, freqs, episode)
    t0     = time.perf_counter()
    think, answer = inference_fn(prompt, temperature)
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

    def fs(s): return sorted(s)

    return {
        # ── identity ──────────────────────────────────────────────────────
        "episode":    episode,
        "parse_fail": False,
        # ── state: full family + features ─────────────────────────────────
        "n":          n,
        "family":     [fs(F) for F in sorted(family, key=lambda s: (len(s), sorted(s)))],
        "family_size": len(family),
        "phi_scores": phi.tolist(),
        "freqs":      freqs,
        "has_singleton": any(len(F) == 1 for F in family),
        "has_empty":     frozenset() in family,
        "density":       len(family) / (2 ** n),
        # ── action ────────────────────────────────────────────────────────
        "element":    x_idx + 1,       # 1-indexed
        "element_0":  x_idx,           # 0-indexed for env replay
        "rule":       RULES[rule_idx],
        "rule_idx":   rule_idx,
        "reason":     reason,
        # ── mapping trace: the actual φ built ─────────────────────────────
        # Each entry: A (bad set), B (chosen partner), image = A∪B
        # This is the data needed to mine new rules later.
        "F_bad":          info.get("F_bad", []),
        "F_good":         info.get("F_good", []),
        "mapping_trace":  info.get("mapping_trace", []),
        # ── outcome ───────────────────────────────────────────────────────
        "reward":    reward,
        "injective": info["injective"],
        "known":     info.get("known", False),
        "coverage":  info["coverage"],
        # ── LLM metadata ──────────────────────────────────────────────────
        "model":     model_name,
        "think":     think,
        "elapsed":   elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,   default=10)
    parser.add_argument("--n",           type=int,   default=4)
    parser.add_argument("--quiet",       action="store_true",
                        help="suppress <think> blocks")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--backend",     choices=["ollama", "hf"], default="ollama",
                        help="ollama (local server) or hf (HuggingFace direct)")
    parser.add_argument("--hf-model",    default=DEFAULT_HF_MODEL,
                        help="HuggingFace model ID (--backend hf only)")
    args = parser.parse_args()

    inference_fn, model_name = build_inference_fn(args.backend, args.hf_model)
    env = FranklEnv(n=args.n, max_steps=1, seed=99)

    results      = []
    rule_counts  = defaultdict(int)
    rule_success = defaultdict(int)

    # ── Log file setup ────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    os.makedirs("traces", exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = model_name.replace("/", "-").replace(":", "-")
    log_path   = f"logs/run_{safe_name}_{ts}_n{args.n}_ep{args.episodes}.txt"
    trace_path = f"traces/trace_{safe_name}_{ts}_n{args.n}_ep{args.episodes}.jsonl"
    log_file   = open(log_path, "w")
    trace_file = open(trace_path, "w")

    def log(line=""):
        print(line)
        log_file.write(line + "\n")

    log(f"Run started : {datetime.now().isoformat()}")
    log(f"Model       : {model_name}  (backend={args.backend})")
    log(f"n           : {args.n}")
    log(f"Episodes    : {args.episodes}")
    log("=" * 70)

    for ep in range(1, args.episodes + 1):
        r = run_episode(env, ep,
                        inference_fn=inference_fn,
                        model_name=model_name,
                        show_think=not args.quiet,
                        temperature=args.temperature)
        results.append(r)
        rule_counts[r["rule"]] += 1
        if r["injective"]:
            rule_success[r["rule"]] += 1

        # Write full trace record (one JSON object per line)
        trace_file.write(json.dumps({k: v for k, v in r.items()
                                     if k != "think"}) + "\n")
        trace_file.flush()

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

    log(f"\nLog saved   : {log_path}")
    log(f"Traces saved: {trace_path}  ({args.episodes} records)")
    log_file.close()
    trace_file.close()


if __name__ == "__main__":
    main()
