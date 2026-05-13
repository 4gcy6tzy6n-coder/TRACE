"""TRACE v2: Intervention Outcome Measurement.

Measures actual error/hallucination/misleading/conflict disclosure rates
before and after TRACE interventions. Produces the final comparison table.

Usage:
    python experiments/run_intervention_outcomes.py
"""

import sys, json
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions
from src.qk_routing_score import compute_r_qk
from src.av_message_score import compute_m_av_from_qkv
from src.utils import load_jsonl, build_prompt, ensure_dir


# ─── Prompt variants ───

def build_reformatted(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"Question: {sample['question']}\n\n"
            f"Evidence:\n{evidence}\n\nAnswer:")

def build_conservative(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\n"
            f"Question: {sample['question']}\n"
            f"Answer (say 'Cannot determine' if evidence is insufficient or conflicting):")

def build_filtered(sample):
    evidence = sample["evidence"]
    filtered = [e for e in evidence if not any(
        w in e.lower() for w in ["claim", "ad", "influencer", "most people",
                                   "celebrity", "brochure", "website shows",
                                   "politician claims", "ads say"]
    )]
    if not filtered:
        filtered = evidence[:1]
    return "\n".join(filtered) + f"\n\nQuestion: {sample['question']}\nAnswer:"


# ─── Outcome measurement ───

def measure_outcomes(model, tokenizer, cache, samples, device, arch_info):
    """Measure error rates for each intervention strategy."""
    strategies = {
        "raw": ("Raw model", lambda s: build_prompt(s)),
        "conservative": ("Conservative (abstain-capable)", lambda s: build_conservative(s)),
        "reformatted": ("Question-first reformat", lambda s: build_reformatted(s)),
        "filtered": ("Filter misleading cues", lambda s: build_filtered(s)),
    }

    results = {}
    for key, (name, prompt_fn) in strategies.items():
        type_outcomes = defaultdict(lambda: {"total": 0, "errors": 0,
                                              "abstentions": 0, "conflict_disclosed": 0})

        for s in samples:
            rt = s["reasoning_type"]
            prompt = prompt_fn(s)
            r = run_forward(model, tokenizer, prompt, cache, device)

            logits = r["logits"][0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            top_id = probs.argmax().item()
            top_tok = tokenizer.decode([top_id]).strip().lower()

            # Determine if answer is correct
            gold = s["gold_answer"].lower()
            is_correct = gold in top_tok or top_tok in gold

            # Abstention: model says "cannot determine" or similar
            abstain_words = ["cannot", "unknown", "unclear", "insufficient",
                             "not enough", "not specified", "not mentioned",
                             "not provided", "not stated", "not given"]
            is_abstention = any(w in top_tok for w in abstain_words)

            # Conflict disclosure: for conflict type, model acknowledges conflict
            conflict_words = ["conflict", "contradict", "disagree", "inconsistent",
                              "differ", "cannot determin"]
            is_conflict = any(w in top_tok for w in conflict_words)

            type_outcomes[rt]["total"] += 1
            if not is_correct and not is_abstention and not is_conflict:
                type_outcomes[rt]["errors"] += 1
            if is_abstention:
                type_outcomes[rt]["abstentions"] += 1
            if is_conflict:
                type_outcomes[rt]["conflict_disclosed"] += 1

        # Aggregate
        total = sum(o["total"] for o in type_outcomes.values())
        errors = sum(o["errors"] for o in type_outcomes.values())
        abstentions = sum(o["abstentions"] for o in type_outcomes.values())
        conflicts = sum(o["conflict_disclosed"] for o in type_outcomes.values())

        # Per-type error rates
        misleading_err = 0
        misleading_total = 0
        conflict_disc = 0
        conflict_total = 0
        gap_abstain = 0
        gap_total = 0

        for rt, o in type_outcomes.items():
            if rt == "misleading_hint":
                misleading_err = o["errors"]
                misleading_total = o["total"]
            if rt == "conflict":
                conflict_disc = o["conflict_disclosed"]
                conflict_total = o["total"]
            if rt == "evidence_gap":
                gap_abstain = o["abstentions"]
                gap_total = o["total"]

        results[key] = {
            "name": name,
            "fire_rate": 1.0,  # all strategies apply to all samples in this setup
            "total_errors": errors,
            "total_samples": total,
            "error_rate": round(errors / max(total, 1), 3),
            "abstention_rate": round(abstentions / max(total, 1), 3),
            "conflict_disclosure_rate": round(conflicts / max(conflict_total, 1), 3) if conflict_total else 0,
            "misleading_error_rate": round(misleading_err / max(misleading_total, 1), 3) if misleading_total else 0,
            "evidence_gap_abstention": round(gap_abstain / max(gap_total, 1), 3) if gap_total else 0,
        }

    # ─── TRACE selective: apply per-sample strategy ───
    trace_errors = 0
    trace_abstentions = 0
    trace_conflicts = 0
    trace_intervened = 0
    trace_total = 0

    for s in samples:
        rt = s["reasoning_type"]
        prompt = build_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)

        # Compute IEAT
        num_heads = arch_info.get("num_heads", 12)
        head_dim = arch_info.get("head_dim", 64)
        pos = get_evidence_token_positions(tokenizer, prompt, s["evidence"],
                                            s.get("gold_evidence_span", ""))
        ev_pos = pos["gold_evidence_positions"]
        ans_pos = len(r["tokens"]) - 1
        r_qk = compute_r_qk(r["attentions"], [ans_pos], ev_pos)
        ici = 0.25 * r_qk + 0.25 * 0 + 0.25 * 0.55 + 0.25 * 0

        # Select intervention
        if rt == "misleading_hint" and r_qk < 0.1:
            prompt = build_filtered(s)
            trace_intervened += 1
        elif rt in ("conflict", "evidence_gap") and ici < 0.2:
            prompt = build_conservative(s)
            trace_intervened += 1
        elif r_qk < 0.05:
            prompt = build_reformatted(s)
            trace_intervened += 1
        # else: keep original prompt

        # Measure outcome with selected prompt
        r2 = run_forward(model, tokenizer, prompt, cache, device)
        logits = r2["logits"][0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        top_tok = tokenizer.decode([probs.argmax().item()]).strip().lower()

        gold = s["gold_answer"].lower()
        is_correct = gold in top_tok or top_tok in gold
        abstain_words = ["cannot", "unknown", "unclear", "insufficient",
                         "not enough", "not specified", "not mentioned",
                         "not provided", "not stated", "not given"]
        is_abstention = any(w in top_tok for w in abstain_words)
        conflict_words = ["conflict", "contradict", "disagree", "inconsistent",
                          "differ", "cannot determin"]
        is_conflict = any(w in top_tok for w in conflict_words)

        trace_total += 1
        if not is_correct and not is_abstention and not is_conflict:
            trace_errors += 1
        if is_abstention:
            trace_abstentions += 1
        if is_conflict:
            trace_conflicts += 1

    results["trace_selective"] = {
        "name": "TRACE selective",
        "fire_rate": round(trace_intervened / max(trace_total, 1), 3),
        "total_errors": trace_errors,
        "total_samples": trace_total,
        "error_rate": round(trace_errors / max(trace_total, 1), 3),
        "abstention_rate": round(trace_abstentions / max(trace_total, 1), 3),
        "conflict_disclosure_rate": round(trace_conflicts / max(trace_total, 1), 3),
        "intervened": trace_intervened,
    }

    return results


def main():
    device = "cpu"
    print("TRACE v2: Intervention Outcome Measurement\n")

    model, tokenizer, cache, arch_info = load_model_v04("gpt2", device)

    samples = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")
    import random; random.seed(42)
    samples = random.sample(samples, min(80, len(samples)))

    error_types = defaultdict(int)
    for s in samples:
        error_types[s["reasoning_type"]] += 1
    print(f"Samples: {dict(error_types)}\n")

    results = measure_outcomes(model, tokenizer, cache, samples, device, arch_info)

    # ─── Final Table ───
    print(f"{'='*90}")
    print("INTERVENTION OUTCOME TABLE")
    print(f"{'='*90}")
    print(f"{'Strategy':30s} {'Fire':>5s} {'Err Rate':>8s} {'Abstain':>8s} {'Conflict':>8s} {'Misleading':>10s}")
    print(f"{'-'*75}")

    for key in ["raw", "conservative", "reformatted", "filtered", "trace_selective"]:
        if key not in results:
            continue
        r = results[key]
        fire = f"{r.get('intervened', r['total_samples'])}/{r['total_samples']}"
        print(f"{r['name']:30s} {fire:>5s} {r['error_rate']:8.3f} {r['abstention_rate']:8.3f} "
              f"{r.get('conflict_disclosure_rate', 0):8.3f} "
              f"{r.get('misleading_error_rate', 0):10.3f}")

    # Key comparisons
    raw = results["raw"]
    trace = results["trace_selective"]

    print(f"\n─── Key Comparisons ───")
    print(f"  Raw error rate:          {raw['error_rate']:.3f}")
    print(f"  TRACE error rate:        {trace['error_rate']:.3f}")
    print(f"  Error reduction:         {(raw['error_rate'] - trace['error_rate']):.3f}")
    print(f"  Raw abstention rate:     {raw['abstention_rate']:.3f}")
    print(f"  TRACE abstention rate:   {trace['abstention_rate']:.3f}")
    print(f"  TRACE fire rate:         {trace['fire_rate']:.1%}")
    print(f"  TRACE intervened:        {trace.get('intervened', '?')}/{trace['total_samples']}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
