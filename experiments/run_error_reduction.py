"""TRACE Error-Reduction Validation — Qwen2.5-0.5B + Controlled + FEVER.

Measures actual error reduction from TRACE-guided intervention vs baselines.
This is the blocking experiment for Nature-level claim.

Usage:
    python experiments/run_error_reduction.py
"""

import sys, json
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions
from src.qk_routing_score import compute_r_qk
from src.utils import load_jsonl, build_prompt, ensure_dir


# ─── Prompt builders ───

def build_filtered(sample):
    evidence = sample["evidence"]
    filtered = [e for e in evidence if not any(
        w in e.lower() for w in ["claim", "ad", "influencer", "most people",
                                   "celebrity", "brochure", "website shows",
                                   "politician claims", "ads say", "marketing",
                                   "sales team claims", "company blog",
                                   "investor pitch claims"]
    )]
    if not filtered:
        filtered = evidence[:1]
    return "\n".join(filtered) + f"\n\nQuestion: {sample['question']}\nAnswer:"

def build_conservative(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\n"
            f"Question: {sample['question']}\n"
            f"Important: if the evidence is insufficient, conflicting, or unclear, "
            f"you must say 'Cannot determine' rather than guessing.\n"
            f"Answer:")


def classify_output(output_text, sample):
    """Classify model output as correct/wrong/abstain/conflict_disclosure."""
    out = output_text.strip().lower()

    gold = sample["gold_answer"].lower()

    # Abstention
    abstain_words = ["cannot determine", "unknown", "unclear", "insufficient",
                     "not enough", "not specified", "not mentioned",
                     "not provided", "not stated", "cannot answer", "undetermined"]
    if any(w in out for w in abstain_words):
        return "abstain"

    # Conflict disclosure
    conflict_words = ["conflict", "contradict", "disagree", "inconsistent",
                      "differ", "cannot determin", "not determin",
                      "different sources", "contradiction"]
    if any(w in out for w in conflict_words):
        return "conflict_disclosure"

    # Correct: gold answer appears in output
    if gold in out or out in gold:
        return "correct"

    # Try token-level partial match for multi-word answers
    gold_tokens = set(gold.split())
    out_tokens = set(out.split())
    if gold_tokens and len(gold_tokens & out_tokens) / len(gold_tokens) > 0.5:
        return "correct"

    return "wrong"


def generate_answer(model, tokenizer, prompt, device, max_new=10):
    """Generate model answer from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated part
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cpu"
    model_id = "Qwen/Qwen2.5-0.5B"

    print(f"TRACE Error-Reduction Validation")
    print(f"Model: {model_id}\n")

    print("Loading model...")
    model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
    num_heads = arch_info.get("num_heads", 14)
    head_dim = arch_info.get("head_dim", 64)
    num_layers = arch_info.get("num_layers", 24)
    family = arch_info.get("family", "unknown")
    print(f"  {arch_info['arch']}: {num_layers}L, {num_heads}H, family={family}")

    # Load samples
    controlled = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")
    fever = load_jsonl(Path(__file__).parent.parent / "data" / "fever_100.jsonl") if Path(
        __file__).parent.parent.joinpath("data/fever_100.jsonl").exists() else []

    # Select error-prone samples
    error_samples = [s for s in controlled
                     if s["reasoning_type"] in ("misleading_hint", "conflict", "evidence_gap")]
    import random; random.seed(42)
    error_samples = random.sample(error_samples, min(40, len(error_samples)))

    print(f"Error-prone samples: {len(error_samples)}")
    by_type = defaultdict(int)
    for s in error_samples:
        by_type[s["reasoning_type"]] += 1
    print(f"  Types: {dict(by_type)}")

    # ─── Run evaluation ───
    strategies = {
        "raw": ("Raw model", lambda s: build_prompt(s)),
        "filtered": ("TRACE: filter misleading", lambda s: build_filtered(s) if s["reasoning_type"] == "misleading_hint" else build_prompt(s)),
        "conservative": ("TRACE: conservative for conflict/gap", lambda s: build_conservative(s) if s["reasoning_type"] in ("conflict", "evidence_gap") else build_prompt(s)),
    }

    all_results = {}
    for key, (name, prompt_fn) in strategies.items():
        print(f"\n─── {name} ───")
        outcomes = defaultdict(int)

        for s in error_samples:
            prompt = prompt_fn(s)
            answer = generate_answer(model, tokenizer, prompt, device, max_new=15)
            result = classify_output(answer, s)
            outcomes[result] += 1

        total = len(error_samples)
        correct = outcomes.get("correct", 0)
        wrong = outcomes.get("wrong", 0)
        abstain = outcomes.get("abstain", 0)
        conflict = outcomes.get("conflict_disclosure", 0)

        # Error rate: wrong answers that aren't abstentions or conflict disclosures
        errors = wrong
        error_rate = errors / max(total, 1)

        print(f"  Correct: {correct}, Wrong: {wrong}, Abstain: {abstain}, Conflict: {conflict}")
        print(f"  Error rate: {error_rate:.3f}")
        print(f"  Safe output rate (abstain+conflict): {(abstain+conflict)/max(total,1):.3f}")

        all_results[key] = {
            "name": name,
            "correct": correct, "wrong": wrong, "abstain": abstain,
            "conflict_disclosure": conflict, "total": total,
            "error_rate": round(error_rate, 3),
            "safe_output_rate": round((abstain + conflict) / max(total, 1), 3),
        }

    # ─── TRACE selective ───
    print(f"\n─── TRACE selective (per-sample strategy) ───")
    trace_outcomes = defaultdict(int)
    trace_intervened = 0

    for s in error_samples:
        rt = s["reasoning_type"]

        # Compute IEAT to select intervention
        prompt = build_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)
        pos = get_evidence_token_positions(tokenizer, prompt, s["evidence"],
                                            s.get("gold_evidence_span", ""))
        ev_pos = pos["gold_evidence_positions"]
        r_qk = compute_r_qk(r["attentions"], [len(r["tokens"]) - 1], ev_pos)

        # Select intervention based on mechanism + error type
        if rt == "misleading_hint" and r_qk < 0.1:
            prompt = build_filtered(s)
            trace_intervened += 1
        elif rt in ("conflict", "evidence_gap") and r_qk < 0.05:
            prompt = build_conservative(s)
            trace_intervened += 1
        # else: keep default

        answer = generate_answer(model, tokenizer, prompt, device, max_new=15)
        result = classify_output(answer, s)
        trace_outcomes[result] += 1

    total = len(error_samples)
    trace_errors = trace_outcomes.get("wrong", 0)
    trace_correct = trace_outcomes.get("correct", 0)
    trace_abstain = trace_outcomes.get("abstain", 0)
    trace_conflict = trace_outcomes.get("conflict_disclosure", 0)

    print(f"  Correct: {trace_correct}, Wrong: {trace_errors}, "
          f"Abstain: {trace_abstain}, Conflict: {trace_conflict}")
    print(f"  Intervened: {trace_intervened}/{total}")
    print(f"  Error rate: {trace_errors/max(total,1):.3f}")

    all_results["trace_selective"] = {
        "name": "TRACE selective",
        "correct": trace_correct, "wrong": trace_errors,
        "abstain": trace_abstain, "conflict_disclosure": trace_conflict,
        "total": total,
        "error_rate": round(trace_errors / max(total, 1), 3),
        "safe_output_rate": round((trace_abstain + trace_conflict) / max(total, 1), 3),
        "intervened": trace_intervened,
        "fire_rate": round(trace_intervened / max(total, 1), 3),
    }

    # ─── Final table ───
    print(f"\n{'='*80}")
    print("TRACE ERROR-REDUCTION VALIDATION — FINAL TABLE")
    print(f"{'='*80}")
    print(f"{'Strategy':30s} {'Error%':>7s} {'Correct':>7s} {'Safe%':>7s} {'Fire%':>7s}")
    print(f"{'-'*55}")

    raw = all_results["raw"]
    for key in ["raw", "filtered", "conservative", "trace_selective"]:
        if key not in all_results:
            continue
        r = all_results[key]
        fire = f"{r.get('fire_rate', 1.0):.1%}" if 'fire_rate' in r else "100%"
        print(f"{r['name']:30s} {r['error_rate']:7.1%} {r['correct']:4d}/{r['total']:3d} "
              f"{r['safe_output_rate']:7.1%} {fire:>7s}")

    # Error reduction
    raw_err = raw["error_rate"]
    trace_err = all_results["trace_selective"]["error_rate"]
    if raw_err > 0:
        reduction = (raw_err - trace_err) / raw_err
        print(f"\n  TRACE error reduction vs raw: {reduction:.1%}")
    else:
        print(f"\n  Raw error rate is 0; no reduction possible to measure.")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "error_reduction_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
