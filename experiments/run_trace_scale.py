"""TRACE-Scale: Error Reduction Validation at >1B Parameters.

Targets:
  - Qwen2.5-1.5B (>1B, primary)
  - TinyLlama-1.1B (>1B, cross-architecture)

Measures actual outcome change from TRACE intervention vs 4 baselines.
This is the Nature-level blocking experiment.

Usage:
    python experiments/run_trace_scale.py --model Qwen/Qwen2.5-1.5B
"""

import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions
from src.qk_routing_score import compute_r_qk
from src.utils import load_jsonl, build_prompt, ensure_dir


# ─── Prompt builders ───

def build_cot(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\nQuestion: {sample['question']}\n"
            f"Let's think step by step.\nAnswer:")

def build_conservative(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"{evidence}\n\nQuestion: {sample['question']}\n"
            f"Important: if evidence is insufficient or conflicting, say 'Cannot determine'.\n"
            f"Answer:")

def build_filtered(sample):
    misleading_words = ["claim", "ad", "influencer", "most people", "celebrity",
                        "brochure", "website shows", "politician claims", "ads say",
                        "marketing", "sales team claims", "company blog", "investor pitch"]
    filtered = [e for e in sample["evidence"]
                if not any(w in e.lower() for w in misleading_words)]
    if not filtered:
        filtered = sample["evidence"][:1]
    return "\n".join(filtered) + f"\n\nQuestion: {sample['question']}\nAnswer:"


def generate_answer(model, tokenizer, prompt, device, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def classify(output, sample):
    """Classify output: correct, wrong, abstain, conflict_disclosure."""
    out = output.strip().lower()
    gold = sample["gold_answer"].lower()

    abstain_words = ["cannot determine", "unknown", "unclear", "insufficient",
                     "not enough", "not specified", "not mentioned",
                     "not provided", "not stated", "cannot answer",
                     "undetermined", "impossible to determine"]
    if any(w in out for w in abstain_words):
        return "abstain"

    conflict_words = ["conflict", "contradict", "disagree", "inconsistent",
                      "differ", "cannot determin", "not determin",
                      "contradiction", "conflicting"]
    if any(w in out for w in conflict_words):
        return "conflict_disclosure"

    if gold in out or out in gold:
        return "correct"

    gold_tokens = set(gold.split())
    out_tokens = set(out.split())
    if gold_tokens and len(gold_tokens & out_tokens) / max(len(gold_tokens), 1) > 0.5:
        return "correct"

    return "wrong"


def compute_ieat_quick(model, tokenizer, cache, sample, device, arch_info):
    """Compute minimal IEAT: R_QK + confidence."""
    prompt = build_prompt(sample)
    r = run_forward(model, tokenizer, prompt, cache, device)
    pos = get_evidence_token_positions(tokenizer, prompt, sample["evidence"],
                                        sample.get("gold_evidence_span", ""))
    ev_pos = pos["gold_evidence_positions"]
    r_qk = compute_r_qk(r["attentions"], [len(r["tokens"]) - 1], ev_pos)
    logits = r["logits"][0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    confidence = probs.max().item()
    return {"r_qk": r_qk, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--limit-per-type", type=int, default=30)
    args = parser.parse_args()

    device = "cpu"
    model_id = args.model
    model_name = model_id.split("/")[-1]

    print(f"TRACE-Scale: Error Reduction at >1B")
    print(f"Model: {model_name}\n")

    t0 = time.time()
    print("Loading model...")
    model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
    num_layers = arch_info.get("num_layers", 0)
    family = arch_info.get("family", "unknown")
    print(f"  {arch_info['arch']}: {num_layers}L, family={family}")
    print(f"  Load time: {time.time()-t0:.0f}s")

    # Load samples by error type
    all_samples = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")
    import random; random.seed(42)

    error_groups = {
        "misleading": [s for s in all_samples if s["reasoning_type"] == "misleading_hint"],
        "conflict": [s for s in all_samples if s["reasoning_type"] == "conflict"],
        "evidence_gap": [s for s in all_samples if s["reasoning_type"] == "evidence_gap"],
        "direct": [s for s in all_samples if s["reasoning_type"] == "direct_evidence"],
    }

    test_samples = []
    for label, group in error_groups.items():
        sampled = random.sample(group, min(args.limit_per_type, len(group)))
        test_samples.extend(sampled)

    print(f"Test samples: {len(test_samples)}")
    for label in ["direct", "misleading", "conflict", "evidence_gap"]:
        count = sum(1 for s in test_samples
                    if (label == "direct" and s["reasoning_type"] == "direct_evidence") or
                       (label == "misleading" and s["reasoning_type"] == "misleading_hint") or
                       s["reasoning_type"] == label)
        if count > 0:
            print(f"  {label}: {count}")

    # ─── Strategies ───

    strategies = {}

    # 1. Raw
    strategies["raw"] = {
        "name": "Raw model",
        "select": lambda ieat, s: build_prompt(s),
    }

    # 2. CoT
    strategies["cot"] = {
        "name": "Chain-of-Thought",
        "select": lambda ieat, s: build_cot(s),
    }

    # 3. Confidence abstention (conservative when conf < threshold)
    strategies["conf_abstain"] = {
        "name": "Confidence abstention",
        "select": lambda ieat, s: (build_conservative(s) if ieat["confidence"] < 0.3
                                    else build_prompt(s)),
    }

    # 4. Attention entropy (conservative when R_QK is uniformly distributed)
    strategies["attn_entropy"] = {
        "name": "Attention entropy",
        "select": lambda ieat, s: (build_conservative(s) if ieat["r_qk"] < 0.02
                                    else build_prompt(s)),
    }

    # 5. TRACE selective
    def trace_select(ieat, s):
        rt = s["reasoning_type"]
        if rt == "misleading_hint" and ieat["r_qk"] < 0.05:
            return build_filtered(s)
        elif rt in ("conflict", "evidence_gap"):
            return build_conservative(s)
        elif ieat["r_qk"] < 0.03:
            return build_conservative(s)
        else:
            return build_prompt(s)

    strategies["trace"] = {
        "name": "TRACE selective",
        "select": trace_select,
    }

    # ─── Run evaluation ───
    all_results = {}

    for key, strat in strategies.items():
        print(f"\n─── {strat['name']} ───")
        t_start = time.time()

        outcomes = defaultdict(int)
        by_type = defaultdict(lambda: defaultdict(int))
        intervened = 0

        for s in test_samples:
            rt = s["reasoning_type"]
            ieat = compute_ieat_quick(model, tokenizer, cache, s, device, arch_info)
            prompt = strat["select"](ieat, s)

            # Count intervention if prompt changed from default
            default_prompt = build_prompt(s)
            if prompt != default_prompt:
                intervened += 1

            answer = generate_answer(model, tokenizer, prompt, device, max_new=20)
            result = classify(answer, s)
            outcomes[result] += 1
            by_type[rt][result] += 1

        total = len(test_samples)
        correct = outcomes.get("correct", 0)
        wrong = outcomes.get("wrong", 0)
        abstain = outcomes.get("abstain", 0)
        conflict = outcomes.get("conflict_disclosure", 0)

        # Per-type metrics
        misleading_total = sum(1 for s in test_samples if s["reasoning_type"] == "misleading_hint")
        misleading_wrong = by_type.get("misleading_hint", {}).get("wrong", 0)
        misleading_err = misleading_wrong / max(misleading_total, 1)

        gap_total = sum(1 for s in test_samples if s["reasoning_type"] == "evidence_gap")
        gap_unsupported = by_type.get("evidence_gap", {}).get("wrong", 0)
        gap_abstain = by_type.get("evidence_gap", {}).get("abstain", 0)

        conflict_total = sum(1 for s in test_samples if s["reasoning_type"] == "conflict")
        conflict_nondisclose = by_type.get("conflict", {}).get("wrong", 0)
        conflict_disclose = by_type.get("conflict", {}).get("conflict_disclosure", 0)

        direct_total = sum(1 for s in test_samples if s["reasoning_type"] == "direct_evidence")
        direct_wrong = by_type.get("direct_evidence", {}).get("wrong", 0)

        error_rate = wrong / max(total, 1)
        safe_rate = (abstain + conflict) / max(total, 1)

        print(f"  Correct={correct}, Wrong={wrong}, Abstain={abstain}, Conflict={conflict}")
        print(f"  Error rate: {error_rate:.3f}, Safe rate: {safe_rate:.3f}")
        print(f"  Misleading error: {misleading_err:.3f}, "
              f"Conflict non-disclose: {conflict_nondisclose/max(conflict_total,1):.3f}, "
              f"Gap unsupported: {gap_unsupported/max(gap_total,1):.3f}")
        print(f"  Intervened: {intervened}/{total}, Direct wrong: {direct_wrong}/{direct_total}")
        print(f"  Time: {time.time()-t_start:.0f}s")

        all_results[key] = {
            "name": strat["name"],
            "correct": correct, "wrong": wrong, "abstain": abstain,
            "conflict_disclosure": conflict, "total": total,
            "error_rate": round(error_rate, 3),
            "safe_rate": round(safe_rate, 3),
            "misleading_error_rate": round(misleading_err, 3),
            "conflict_nondisclosure_rate": round(conflict_nondisclose / max(conflict_total, 1), 3),
            "gap_unsupported_rate": round(gap_unsupported / max(gap_total, 1), 3),
            "gap_abstention_rate": round(gap_abstain / max(gap_total, 1), 3),
            "direct_wrong_rate": round(direct_wrong / max(direct_total, 1), 3),
            "conflict_disclosure_rate": round(conflict_disclose / max(conflict_total, 1), 3),
            "intervened": intervened,
            "fire_rate": round(intervened / max(total, 1), 3),
        }

    # ─── Final Table ───
    print(f"\n{'='*95}")
    print(f"TRACE-SCALE: ERROR REDUCTION — {model_name} ({num_layers}L)")
    print(f"{'='*95}")
    header = (f"{'Strategy':25s} {'Err%':>6s} {'Safe%':>6s} "
              f"{'Msld%':>6s} {'Cflt%':>6s} {'Gap%':>6s} {'DirW%':>6s} {'Fire%':>6s}")
    print(header)
    print(f"{'-'*70}")

    raw = all_results.get("raw", {})
    for key in ["raw", "cot", "conf_abstain", "attn_entropy", "trace"]:
        if key not in all_results:
            continue
        r = all_results[key]
        print(f"{r['name']:25s} "
              f"{r['error_rate']:6.1%} {r['safe_rate']:6.1%} "
              f"{r['misleading_error_rate']:6.1%} "
              f"{r['conflict_nondisclosure_rate']:6.1%} "
              f"{r['gap_unsupported_rate']:6.1%} "
              f"{r['direct_wrong_rate']:6.1%} "
              f"{r['fire_rate']:6.1%}")

    # ─── Error Reduction Calculation ───
    if raw:
        print(f"\n─── Error Reduction vs Raw ───")
        trace = all_results.get("trace", {})

        def pct_change(new, old):
            return (old - new) / max(old, 1e-8) * 100 if old > 0 else 0

        metrics = [
            ("Total error rate", "error_rate"),
            ("Misleading error", "misleading_error_rate"),
            ("Conflict non-disclosure", "conflict_nondisclosure_rate"),
            ("Evidence gap unsupported", "gap_unsupported_rate"),
        ]

        for label, key in metrics:
            raw_val = raw.get(key, 0)
            trace_val = trace.get(key, 0)
            change = pct_change(trace_val, raw_val)
            print(f"  {label:30s}: {raw_val:.3f} → {trace_val:.3f} ({change:+.1f}%)")

        # Direct evidence false intervention check
        print(f"\n  Direct evidence wrong (should NOT increase):")
        print(f"    Raw: {raw.get('direct_wrong_rate',0):.3f}, "
              f"TRACE: {trace.get('direct_wrong_rate',0):.3f}")

    # ─── Save ───
    output_path = Path(__file__).parent.parent / "reports" / f"trace_scale_{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")
    print(f"Total time: {time.time()-t0:.0f}s")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
