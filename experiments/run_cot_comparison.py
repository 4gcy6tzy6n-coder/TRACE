"""Compare ICI scores with and without chain-of-thought prompting.

Hypothesis: CoT should increase ICI if internal reasoning is real,
or reveal that CoT is surface-level if ICI doesn't change.

Usage:
    python experiments/run_cot_comparison.py
    python experiments/run_cot_comparison.py --limit 20
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk
from src.av_message_score import compute_m_av_from_qkv
from src.residual_state_score import (
    extract_features_from_last_token,
    train_linear_probe,
    compute_s_x,
    reasoning_type_to_label,
)
from src.ici_calculator import compute_ici_for_sample, save_results
from src.utils import (
    load_jsonl, build_cot_prompt, build_no_cot_prompt, ensure_dir,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def compute_global_s_x_for_samples(model, tokenizer, cache, samples, prompt_fn, device):
    """Compute global S_X for a set of samples with a given prompt builder."""
    features, labels = [], []
    for s in samples:
        prompt = prompt_fn(s)
        label = reasoning_type_to_label(s["reasoning_type"])
        if label < 0:
            continue
        result = run_forward(model, tokenizer, prompt, cache, device)
        feat = extract_features_from_last_token(result["hidden_states"], layer=-1)
        if feat.shape[0] > 0:
            features.append(feat.flatten())
            labels.append(label)

    if len(features) < 2:
        return 0.0
    features = np.array(features)
    labels = np.array(labels)
    num_classes = len(set(labels))
    if num_classes < 2:
        return 0.0

    clf = train_linear_probe(features, labels)
    min_count = min(np.bincount(labels))
    n_splits = min(5, min_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
        acc = cv_scores.mean()
    else:
        acc = clf.score(features, labels)
    return compute_s_x(acc, num_classes)


def evaluate_with_prompt(model, tokenizer, cache, samples, prompt_fn, s_x, device):
    """Run ICI evaluation for all samples with a given prompt function."""
    results = []
    for s in samples:
        prompt = prompt_fn(s)
        result = run_forward(model, tokenizer, prompt, cache, device)

        pos_info = get_evidence_token_positions(
            tokenizer, prompt, s["evidence"], s["gold_evidence_span"]
        )
        ev_pos = pos_info["gold_evidence_positions"]
        answer_positions = find_answer_position(tokenizer, result["tokens"], s["gold_answer"])
        if not answer_positions:
            answer_positions = [len(result["tokens"]) - 1]

        # R_QK
        r_qk = compute_r_qk(result["attentions"], answer_positions, ev_pos)

        # True M_AV
        ans_token_id = tokenizer.encode(s["gold_answer"], add_special_tokens=False)[0]
        m_av_result = compute_m_av_from_qkv(
            model, result["q_per_layer"], result["k_per_layer"],
            result["v_per_layer"], ev_pos, answer_positions[0], ans_token_id
        )
        m_av = m_av_result["M_AV"]

        # C_do skipped for speed in CoT comparison
        ici = compute_ici_for_sample(s["id"], r_qk, m_av, s_x, 0.0)
        ici["reasoning_type"] = s["reasoning_type"]
        ici["label"] = s["label"]
        results.append(ici)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    samples = load_jsonl(
        Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    )
    if args.limit > 0:
        import random
        random.seed(42)
        samples = random.sample(samples, min(args.limit, len(samples)))
    print(f"Evaluating {len(samples)} samples")

    # Compute S_X for both prompt types
    print("\n--- S_X (no-CoT) ---")
    s_x_no_cot = compute_global_s_x_for_samples(model, tokenizer, cache, samples, build_no_cot_prompt, device)
    print(f"S_X (no-CoT): {s_x_no_cot:.4f}")

    print("\n--- S_X (CoT) ---")
    s_x_cot = compute_global_s_x_for_samples(model, tokenizer, cache, samples, build_cot_prompt, device)
    print(f"S_X (CoT):     {s_x_cot:.4f}")

    # Evaluate both conditions
    print("\n--- Evaluating no-CoT ---")
    no_cot_results = evaluate_with_prompt(model, tokenizer, cache, samples, build_no_cot_prompt, s_x_no_cot, device)

    print("\n--- Evaluating CoT ---")
    cot_results = evaluate_with_prompt(model, tokenizer, cache, samples, build_cot_prompt, s_x_cot, device)

    # Compare
    print("\n--- CoT vs no-CoT Comparison ---")
    comparisons = []
    for nr, cr in zip(no_cot_results, cot_results):
        ici_diff = cr["ICI"] - nr["ICI"]
        comparisons.append({
            "sample_id": nr["sample_id"],
            "reasoning_type": nr["reasoning_type"],
            "ICI_no_cot": nr["ICI"],
            "ICI_cot": cr["ICI"],
            "ICI_diff": round(ici_diff, 4),
            "R_QK_no_cot": nr["R_QK"],
            "R_QK_cot": cr["R_QK"],
            "M_AV_no_cot": nr["M_AV"],
            "M_AV_cot": cr["M_AV"],
        })

    # Summary by type
    print(f"\n{'Type':20s} {'n':>3s} {'ICI(no-CoT)':>12s} {'ICI(CoT)':>12s} {'ΔICI':>8s}")
    print("-" * 58)
    type_groups = {}
    for c in comparisons:
        rt = c["reasoning_type"]
        if rt not in type_groups:
            type_groups[rt] = []
        type_groups[rt].append(c)

    for rt, group in type_groups.items():
        avg_no = sum(g["ICI_no_cot"] for g in group) / len(group)
        avg_cot = sum(g["ICI_cot"] for g in group) / len(group)
        avg_diff = avg_cot - avg_no
        print(f"{rt:20s} {len(group):3d} {avg_no:12.4f} {avg_cot:12.4f} {avg_diff:+8.4f}")

    # Save
    output_dir = ensure_dir(Path(__file__).parent.parent / "reports")
    output_path = output_dir / "cot_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    # Summary
    diffs = [c["ICI_diff"] for c in comparisons]
    summary = {
        "num_samples": len(comparisons),
        "avg_ICI_no_cot": round(sum(c["ICI_no_cot"] for c in comparisons) / len(comparisons), 4),
        "avg_ICI_cot": round(sum(c["ICI_cot"] for c in comparisons) / len(comparisons), 4),
        "mean_diff": round(sum(diffs) / len(diffs), 4),
        "by_type": {
            rt: {
                "avg_no_cot": round(sum(g["ICI_no_cot"] for g in group) / len(group), 4),
                "avg_cot": round(sum(g["ICI_cot"] for g in group) / len(group), 4),
            }
            for rt, group in type_groups.items()
        },
        "S_X_no_cot": s_x_no_cot,
        "S_X_cot": s_x_cot,
    }
    with open(output_dir / "cot_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"Summary saved to {output_dir / 'cot_comparison_summary.json'}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
