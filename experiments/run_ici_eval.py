"""Run full ICI evaluation on the toy dataset.

Computes R_QK, M_AV, S_X, C_do for each sample and calculates the ICI score.

v0.2: True M_AV from QKV, per-head analysis, layer-wise S_X, CoT option.

Usage:
    python experiments/run_ici_eval.py
    python experiments/run_ici_eval.py --limit 10
    python experiments/run_ici_eval.py --v2          # use true M_AV (QKV-based)
    python experiments/run_ici_eval.py --heatmap      # export per-head heatmap data
    python experiments/run_ici_eval.py --cot          # use chain-of-thought prompt
    python experiments/run_ici_eval.py --layerwise-sx # per-layer S_X probes
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
from src.qk_routing_score import compute_r_qk, compute_r_qk_per_head, identify_evidence_routing_heads
from src.av_message_score import compute_m_av_proxy, compute_m_av_from_qkv
from src.residual_state_score import (
    extract_features_from_last_token,
    train_linear_probe,
    train_layerwise_probes,
    compute_s_x,
    reasoning_type_to_label,
)
from src.causal_intervention import compute_c_do
from src.ici_calculator import compute_ici_for_sample, save_results
from src.utils import (
    load_jsonl, build_prompt, build_cot_prompt,
    export_heatmap_data, ensure_dir,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def compute_global_s_x(model, tokenizer, cache, samples, prompt_fn, device):
    """Train probe with cross-validation to get honest S_X."""
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
        accuracy = cv_scores.mean()
    else:
        accuracy = clf.score(features, labels)
    return compute_s_x(accuracy, num_classes)


def main():
    parser = argparse.ArgumentParser(description="Run ICI evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--skip-cdo", action="store_true", help="Skip causal intervention (slow)")
    parser.add_argument("--v2", action="store_true", help="Use true M_AV from QKV (v0.2)")
    parser.add_argument("--heatmap", action="store_true", help="Export per-head heatmap data")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompt")
    parser.add_argument("--layerwise-sx", action="store_true", help="Compute per-layer S_X probes")
    parser.add_argument("--output", type=str, default="reports/ici_results.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"v0.2 features: v2={args.v2}, heatmap={args.heatmap}, cot={args.cot}, layerwise-sx={args.layerwise_sx}")

    # Prompt function
    prompt_fn = build_cot_prompt if args.cot else build_prompt

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)
    if args.limit > 0:
        import random
        random.seed(42)
        samples = random.sample(samples, min(args.limit, len(samples)))
    print(f"Evaluating {len(samples)} samples")

    # Compute global S_X
    print("\n--- Computing S_X (probe training) ---")
    global_s_x = compute_global_s_x(model, tokenizer, cache, samples, prompt_fn, device)
    print(f"Global S_X: {global_s_x:.4f}")

    # Per-layer S_X (optional)
    layerwise_s_x = {}
    if args.layerwise_sx:
        print("\n--- Computing layer-wise S_X ---")
        all_hs = []
        all_labels = []
        for s in samples:
            prompt = prompt_fn(s)
            label = reasoning_type_to_label(s["reasoning_type"])
            if label < 0:
                continue
            result = run_forward(model, tokenizer, prompt, cache, device)
            all_hs.append(result["hidden_states"])
            all_labels.append(label)
        all_labels_arr = np.array(all_labels)
        layerwise_s_x = train_layerwise_probes(all_hs, all_labels_arr, num_layers=12)
        for l in range(12):
            info = layerwise_s_x[l]
            print(f"  Layer {l:2d}: acc={info['accuracy']:.4f}, S_X={info['S_X']:.4f}")

    # Evaluate each sample
    print("\n--- Computing per-sample scores ---")
    results = []

    # For heatmap: aggregate per-head scores across samples
    all_per_head_r_qk: dict[int, dict[int, list[float]]] = {}
    all_per_head_m_av: dict[int, dict[int, list[float]]] = {}

    for idx, sample in enumerate(samples):
        prompt = prompt_fn(sample)
        print(f"\n[{idx + 1}/{len(samples)}] {sample['id']} ({sample['reasoning_type']})")

        result = run_forward(model, tokenizer, prompt, cache, device)

        pos_info = get_evidence_token_positions(
            tokenizer, prompt, sample["evidence"], sample["gold_evidence_span"]
        )
        evidence_positions = pos_info["gold_evidence_positions"]
        answer_positions = find_answer_position(
            tokenizer, result["tokens"], sample["gold_answer"]
        )
        if not answer_positions:
            answer_positions = [len(result["tokens"]) - 1]

        # --- R_QK ---
        r_qk = compute_r_qk(result["attentions"], answer_positions, evidence_positions)
        print(f"  R_QK: {r_qk:.4f}")

        # Per-head R_QK
        if args.heatmap:
            per_head = compute_r_qk_per_head(result["attentions"], answer_positions, evidence_positions)
            for l, heads in per_head.items():
                if l not in all_per_head_r_qk:
                    all_per_head_r_qk[l] = {}
                for h, score in heads.items():
                    if h not in all_per_head_r_qk[l]:
                        all_per_head_r_qk[l][h] = []
                    all_per_head_r_qk[l][h].append(score)

        # --- M_AV ---
        if args.v2:
            # True M_AV from QKV
            ans_token_id = tokenizer.encode(sample["gold_answer"], add_special_tokens=False)[0]
            m_av_result = compute_m_av_from_qkv(
                model, result["q_per_layer"], result["k_per_layer"],
                result["v_per_layer"], evidence_positions,
                answer_positions[0], ans_token_id,
            )
            m_av = m_av_result["M_AV"]
            print(f"  M_AV (true): {m_av:.4f}")

            if args.heatmap:
                per_head_m = m_av_result["per_head_contributions"]
                for l, heads in per_head_m.items():
                    if l not in all_per_head_m_av:
                        all_per_head_m_av[l] = {}
                    for h, contrib in heads.items():
                        if h not in all_per_head_m_av[l]:
                            all_per_head_m_av[l][h] = []
                        all_per_head_m_av[l][h].append(contrib)
        else:
            # v0.1 proxy
            m_av_result = compute_m_av_proxy(
                model, tokenizer, cache, prompt,
                sample["gold_evidence_span"], sample["gold_answer"], device
            )
            m_av = m_av_result["M_AV"]
            print(f"  M_AV (proxy): {m_av:.4f}")

        # --- S_X (use global) ---
        s_x = global_s_x

        # --- C_do ---
        if args.skip_cdo:
            c_do = 0.0
            print("  C_do: skipped")
        else:
            c_do_result = compute_c_do(
                model, tokenizer, cache, prompt,
                sample["gold_evidence_span"],
                evidence_positions,
                answer_positions,
                sample["gold_answer"],
                device,
            )
            c_do = c_do_result["C_do"]
            print(f"  C_do: {c_do:.4f}")

        # --- ICI ---
        ici_result = compute_ici_for_sample(
            sample["id"], r_qk, m_av, s_x, c_do
        )
        ici_result["reasoning_type"] = sample["reasoning_type"]
        ici_result["label"] = sample["label"]
        ici_result["gold_answer"] = sample["gold_answer"]
        ici_result["evidence_positions"] = evidence_positions
        ici_result["answer_positions"] = answer_positions
        results.append(ici_result)
        print(f"  ICI: {ici_result['ICI']:.4f}")

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    save_results(results, output_path)
    print(f"\nResults saved to {output_path}")

    # Summary by reasoning type
    print("\n--- Summary by Reasoning Type ---")
    type_groups = {}
    for r in results:
        rt = r["reasoning_type"]
        if rt not in type_groups:
            type_groups[rt] = []
        type_groups[rt].append(r)

    for rt, group in type_groups.items():
        avg_ici = sum(g["ICI"] for g in group) / len(group)
        avg_rqk = sum(g["R_QK"] for g in group) / len(group)
        print(f"  {rt:20s}: n={len(group):2d}, ICI={avg_ici:.4f}, R_QK={avg_rqk:.4f}")

    # Export heatmap data
    if args.heatmap:
        # Average per-head scores across samples
        avg_r_qk: dict[int, dict[int, float]] = {}
        avg_m_av: dict[int, dict[int, float]] = {}
        for l in range(12):
            avg_r_qk[l] = {}
            avg_m_av[l] = {}
            for h in range(12):
                rqk_list = all_per_head_r_qk.get(l, {}).get(h, [0.0])
                mav_list = all_per_head_m_av.get(l, {}).get(h, [0.0])
                avg_r_qk[l][h] = sum(rqk_list) / len(rqk_list) if rqk_list else 0.0
                avg_m_av[l][h] = sum(mav_list) / len(mav_list) if mav_list else 0.0

        lw_sx = {layer: info["S_X"] for layer, info in layerwise_s_x.items()}
        heatmap_path = Path(__file__).parent.parent / "reports" / "heatmap_data.json"
        export_heatmap_data(avg_r_qk, avg_m_av, lw_sx, heatmap_path)
        print(f"\nHeatmap data exported to {heatmap_path}")

        # Print evidence-routing heads
        routing_heads = identify_evidence_routing_heads(avg_r_qk, threshold_percentile=80)
        print(f"\nTop evidence-routing heads (R_QK):")
        for l, h, score in routing_heads[:5]:
            print(f"  Layer {l:2d}, Head {h:2d}: {score:.4f}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
