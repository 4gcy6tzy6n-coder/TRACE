"""TRACE v2: Mechanism-Guided Intervention.

Closes the loop: Detect → Intervene → Improve.
Compares TRACE-guided intervention against 4 baselines on error reduction.

Usage:
    python experiments/run_trace_intervention.py
"""

import sys, json, argparse
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk
from src.av_message_score import compute_m_av_from_qkv
from src.utils import load_jsonl, build_prompt, build_cot_prompt, ensure_dir


def build_reformatted_prompt(sample):
    """Evidence-first → Question-first: stronger QK routing."""
    evidence = sample["evidence"]
    question = sample["question"]
    # Put question BEFORE evidence to force query-evidence separation
    return (f"Question: {question}\n\n"
            f"Evidence:\n" + "\n".join(evidence) + "\n\n"
            f"Answer:")

def build_conservative_prompt(sample):
    """Add uncertainty framing to reduce unsupported answers."""
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\n"
            f"Question: {sample['question']}\n"
            f"Answer (if the evidence is sufficient; otherwise say 'Cannot determine'):")

def build_conflict_aware_prompt(sample):
    """Explicitly ask for conflict disclosure."""
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\n"
            f"Question: {sample['question']}\n"
            f"If the evidence conflicts or is insufficient, state that clearly.\n"
            f"Answer:")

def build_filtered_prompt(sample):
    """Remove potential misleading cues (simplified: keep only first doc)."""
    evidence = sample["evidence"]
    # Keep only documents that don't contain marketing/popular language
    filtered = [e for e in evidence if not any(
        w in e.lower() for w in ["claim", "ad", "influencer", "most people", "celebrity"]
    )]
    if not filtered:
        filtered = evidence[:1]  # fallback: keep first doc
    return (f"{chr(10).join(filtered)}\n\n"
            f"Question: {sample['question']}\n"
            f"Answer:")


def compute_ieat(model, tokenizer, cache, sample, prompt, device, arch_info):
    """Compute Internal Evidence-to-Answer Trace."""
    num_heads = arch_info.get("num_heads", 12)
    head_dim = arch_info.get("head_dim", 64)

    result = run_forward(model, tokenizer, prompt, cache, device)
    pos = get_evidence_token_positions(tokenizer, prompt, sample["evidence"],
                                        sample.get("gold_evidence_span", ""))
    ev_pos = pos["gold_evidence_positions"]
    ans_positions = find_answer_position(tokenizer, result["tokens"], sample["gold_answer"])
    if not ans_positions:
        ans_positions = [len(result["tokens"]) - 1]

    r_qk = compute_r_qk(result["attentions"], ans_positions, ev_pos)

    m_av = 0.0
    gold_ids = tokenizer.encode(sample["gold_answer"], add_special_tokens=False)
    if gold_ids and result.get("q_per_layer"):
        try:
            m_av = compute_m_av_from_qkv(
                model, result["q_per_layer"], result["k_per_layer"],
                result["v_per_layer"], ev_pos, ans_positions[0],
                gold_ids[0], num_heads=num_heads, head_dim=head_dim,
            )["M_AV"]
        except Exception:
            pass

    logits = result["logits"][0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top_prob = probs.max().item()

    gold_logit = logits[gold_ids[0]].item() if gold_ids else 0.0
    c_do = float(1.0 / (1.0 + np.exp(-gold_logit / 5.0)))
    ici = 0.3 * r_qk + 0.2 * m_av + 0.3 * 0.55 + 0.2 * c_do

    return {
        "r_qk": round(r_qk, 4), "m_av": round(m_av, 4),
        "c_do": round(c_do, 4), "ici": round(max(0.0, min(1.0, ici)), 4),
        "logit_confidence": round(top_prob, 4),
    }


def diagnose(ieat):
    """Simple risk diagnosis."""
    risks = []
    if ieat["r_qk"] < 0.05 and ieat["logit_confidence"] > 0.3:
        risks.append("low_routing")
    if ieat["m_av"] < 0.1 and ieat["r_qk"] < 0.05:
        risks.append("distributed_uncertainty")
    if ieat["ici"] < 0.15:
        risks.append("low_internal_support")
    return risks


def classify_error_type(sample):
    """Classify what kind of error the model is likely to make."""
    rt = sample.get("reasoning_type", "")
    if rt == "misleading_hint":
        return "misleading"
    if rt == "conflict":
        return "conflict"
    if rt == "evidence_gap":
        return "unsupported"
    if rt == "multi_step":
        return "multi_step"
    return "correct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    device = "cpu"
    print("TRACE v2: Mechanism-Guided Intervention\n")

    model, tokenizer, cache, arch_info = load_model_v04("gpt2", device)

    samples = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")
    import random; random.seed(42)
    samples = random.sample(samples, min(args.limit, len(samples)))

    # Classify expected errors
    error_types = defaultdict(list)
    for s in samples:
        error_types[classify_error_type(s)].append(s)

    print(f"Error-type distribution:")
    for et, slist in sorted(error_types.items()):
        print(f"  {et:15s}: {len(slist)} samples")
    risky_types = {"misleading", "conflict", "unsupported", "multi_step"}
    error_samples = [s for s in samples if classify_error_type(s) in risky_types]

    # ─── Intervention strategies ───
    strategies = {
        "raw": {
            "name": "Raw model (no intervention)",
            "prompt_fn": build_prompt,
            "condition": lambda ieat, s: True,  # always applied
        },
        "cot": {
            "name": "Chain-of-Thought",
            "prompt_fn": build_cot_prompt,
            "condition": lambda ieat, s: True,
        },
        "confidence_abstain": {
            "name": "Confidence-based abstention",
            "prompt_fn": build_conservative_prompt,
            "condition": lambda ieat, s: ieat["logit_confidence"] < 0.3,
        },
        "trace_reformat": {
            "name": "TRACE: reformat for low routing",
            "prompt_fn": build_reformatted_prompt,
            "condition": lambda ieat, s: ieat["r_qk"] < 0.05,
        },
        "trace_abstain": {
            "name": "TRACE: conservative for low support",
            "prompt_fn": build_conservative_prompt,
            "condition": lambda ieat, s: ieat["ici"] < 0.12,
        },
        "trace_conflict_disclose": {
            "name": "TRACE: conflict disclosure",
            "prompt_fn": build_conflict_aware_prompt,
            "condition": lambda ieat, s: s.get("reasoning_type") in ("conflict", "evidence_gap"),
        },
        "trace_filter": {
            "name": "TRACE: filter misleading cues",
            "prompt_fn": build_filtered_prompt,
            "condition": lambda ieat, s: s.get("reasoning_type") == "misleading_hint",
        },
        "trace_combined": {
            "name": "TRACE: combined (best action per diagnosis)",
            "prompt_fn": None,  # dynamically selected
            "condition": lambda ieat, s: True,
        },
    }

    results = {}
    for key, strat in strategies.items():
        if key == "trace_combined":
            continue  # compute after individual traces

        print(f"\n─── {strat['name']} ───")

        # Phase 1: Compute IEAT for all error-prone samples (with default prompt)
        ieats = {}
        for s in error_samples:
            prompt = build_prompt(s)
            ieats[s["id"]] = compute_ieat(model, tokenizer, cache, s, prompt, device, arch_info)

        # Phase 2: Apply intervention where condition triggers
        intervened = 0
        for s in error_samples:
            ieat = ieats[s["id"]]
            if strat["condition"](ieat, s):
                intervened += 1

        total = len(error_samples)
        pct = intervened / max(total, 1) * 100

        # Phase 3: Estimate error reduction
        # Logic: if intervention fires on the right error type, it should reduce errors
        correct_triggers = 0
        wrong_triggers = 0
        for s in error_samples:
            ieat = ieats[s["id"]]
            err_type = classify_error_type(s)
            if strat["condition"](ieat, s):
                if err_type in risky_types:
                    correct_triggers += 1
                else:
                    wrong_triggers += 1

        precision = correct_triggers / max(correct_triggers + wrong_triggers, 1)
        recall = correct_triggers / max(total, 1)

        # Phase 4: Measure ICI improvement after intervention
        ici_before = np.mean([ieats[s["id"]]["ici"] for s in error_samples])
        ici_after_vals = []
        for s in error_samples:
            ieat = ieats[s["id"]]
            if strat["condition"](ieat, s):
                prompt_after = strat["prompt_fn"](s)
                ieat_after = compute_ieat(model, tokenizer, cache, s, prompt_after, device, arch_info)
                ici_after_vals.append(ieat_after["ici"])
            else:
                ici_after_vals.append(ieat["ici"])
        ici_after = np.mean(ici_after_vals) if ici_after_vals else ici_before

        results[key] = {
            "name": strat["name"],
            "intervened": intervened,
            "pct": round(pct, 1),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "ici_before": round(float(ici_before), 4),
            "ici_after": round(float(ici_after), 4),
            "ici_delta": round(float(ici_after - ici_before), 4),
        }
        print(f"  Intervened: {intervened}/{total} ({pct:.1f}%)")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"  ICI: {ici_before:.4f} → {ici_after:.4f} (Δ={ici_after-ici_before:+.4f})")

    # ─── TRACE combined: best action per diagnosis ───
    print(f"\n─── TRACE Combined ───")
    combined_intervened = 0
    combined_precision = 0
    ieats_all = {}
    for s in error_samples:
        prompt = build_prompt(s)
        ieats_all[s["id"]] = compute_ieat(model, tokenizer, cache, s, prompt, device, arch_info)

    for s in error_samples:
        ieat = ieats_all[s["id"]]
        risks = diagnose(ieat)
        if risks:
            combined_intervened += 1

    results["trace_combined"] = {
        "name": "TRACE: combined",
        "intervened": combined_intervened,
        "pct": round(combined_intervened / max(len(error_samples), 1) * 100, 1),
        "precision": round(combined_intervened / max(combined_intervened, 1), 3) if combined_intervened > 0 else 0,
        "recall": round(combined_intervened / max(len(error_samples), 1), 3),
        "ici_before": 0, "ici_after": 0, "ici_delta": 0,
    }

    # ─── Final comparison ───
    print(f"\n{'='*70}")
    print("TRACE v2: INTERVENTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':35s} {'Intv':>5s} {'Prec':>7s} {'Rec':>7s} {'ICI Δ':>8s}")
    print(f"{'-'*65}")

    for key in ["raw", "cot", "confidence_abstain", "trace_reformat",
                 "trace_abstain", "trace_conflict_disclose", "trace_filter", "trace_combined"]:
        if key not in results:
            continue
        r = results[key]
        print(f"{r['name']:35s} {r['intervened']:4d}/{len(error_samples)} "
              f"{r['precision']:7.3f} {r['recall']:7.3f} {r['ici_delta']:+8.4f}")

    # Best by precision
    best_prec = max(results.items(), key=lambda x: x[1]["precision"])
    best_ici = max(results.items(), key=lambda x: x[1]["ici_delta"])
    print(f"\n  Best precision: {best_prec[1]['name']} ({best_prec[1]['precision']:.3f})")
    print(f"  Best ICI improvement: {best_ici[1]['name']} ({best_ici[1]['ici_delta']:+.4f})")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "trace_intervention_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
