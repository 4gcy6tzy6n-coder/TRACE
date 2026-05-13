"""v0.5: Scale-aware ICI calibration and distributed contribution analysis.

Runs full v0.5 pipeline:
1. Within-model R_QK gap/ratio
2. Scale-aware weight calibration per model
3. Distributed pathway contribution (attention/MLP/residual)
4. Fixed vs calibrated ICI comparison

Usage:
    python experiments/run_v05_calibration.py
    python experiments/run_v05_calibration.py --model gpt2,gpt2-medium
"""

import sys, json, argparse
from pathlib import Path

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk, compute_r_qk_per_head
from src.av_message_score import compute_m_av_from_qkv, compute_m_av_proxy
from src.residual_state_score import (
    extract_features_from_last_token, train_linear_probe,
    compute_s_x, reasoning_type_to_label,
)
from src.scale_aware_ici import (
    compute_within_model_rqk_gap, compute_scale_aware_weights,
    calibrate_weights_from_runs, compute_scale_aware_ici,
    compare_fixed_vs_calibrated,
)
from src.distributed_mav import (
    compute_distributed_contribution, compute_pathway_switch_score,
)
from src.ici_calculator import compute_ici_for_sample
from src.utils import load_jsonl, build_prompt, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score


MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
}


def evaluate_model_v05(model_name, model_id, samples, device, limit=30):
    """Full v0.5 evaluation per model: R_QK, M_AV, S_X, pathway analysis."""
    print(f"\n{'='*60}")
    print(f"v0.5: {model_name} ({model_id})")
    print(f"{'='*60}")

    model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
    num_heads = arch_info.get("num_heads", 12)
    head_dim = arch_info.get("head_dim", 64)
    num_layers = arch_info.get("num_layers", 12)

    import random; random.seed(42)
    eval_samples = random.sample(samples, min(limit, len(samples)))

    # Collect per-type data
    per_type_rqk = {}
    per_type_mav = {}
    per_type_ici_fixed = {}
    features, labels = [], []
    distributed_results = []

    for s in eval_samples:
        prompt = build_prompt(s)
        result = run_forward(model, tokenizer, prompt, cache, device)

        # Positions
        pos_info = get_evidence_token_positions(tokenizer, prompt, s["evidence"], s["gold_evidence_span"])
        ev_pos = pos_info["gold_evidence_positions"]
        ans_positions = find_answer_position(tokenizer, result["tokens"], s["gold_answer"])
        if not ans_positions:
            ans_positions = [len(result["tokens"]) - 1]

        # R_QK
        r_qk = compute_r_qk(result["attentions"], ans_positions, ev_pos)

        # M_AV (proxy for reliability)
        gold_token_ids = tokenizer.encode(s["gold_answer"], add_special_tokens=False)
        m_av = 0.0
        if gold_token_ids and result.get("q_per_layer"):
            try:
                m_av_result = compute_m_av_from_qkv(
                    model, result["q_per_layer"], result["k_per_layer"],
                    result["v_per_layer"], ev_pos, ans_positions[0],
                    gold_token_ids[0], num_heads=num_heads, head_dim=head_dim,
                )
                m_av = m_av_result["M_AV"]
            except Exception:
                try:
                    m_av = compute_m_av_proxy(model, tokenizer, cache, prompt,
                                              s["gold_evidence_span"], s["gold_answer"], device)["M_AV"]
                except Exception:
                    m_av = 0.0

        # Per-type aggregation
        rt = s["reasoning_type"]
        if rt not in per_type_rqk:
            per_type_rqk[rt] = []
            per_type_mav[rt] = []
            per_type_ici_fixed[rt] = []
        per_type_rqk[rt].append(r_qk)
        per_type_mav[rt].append(m_av)
        per_type_ici_fixed[rt].append(0.25 * r_qk + 0.25 * m_av)  # simplified

        # S_X probe features
        label = reasoning_type_to_label(rt)
        if label >= 0:
            # Find deepest clean layer
            clean_layer = -1
            for l in range(len(result["hidden_states"]) - 1, -1, -1):
                if not torch.isnan(result["hidden_states"][l][0][-1]).any():
                    clean_layer = l
                    break
            feat = extract_features_from_last_token(result["hidden_states"], layer=clean_layer)
            if feat.shape[0] > 0 and not np.isnan(feat).any():
                features.append(feat.flatten().astype(np.float32))
                labels.append(label)

        # Distributed pathway (first 10 samples only for speed)
        if len(distributed_results) < 10 and result.get("q_per_layer") and gold_token_ids:
            try:
                dist = compute_distributed_contribution(
                    model, result["q_per_layer"], result["k_per_layer"],
                    result["v_per_layer"], result["mlp_outputs"],
                    result["hidden_states"], ev_pos, ans_positions[0],
                    gold_token_ids[0], num_heads=num_heads, head_dim=head_dim,
                )
                distributed_results.append(dist)
            except Exception:
                pass

    # S_X probe
    features_arr = np.array(features)
    labels_arr = np.array(labels)
    num_classes = len(set(labels_arr))
    if num_classes >= 2 and len(features_arr) >= 10:
        clf = train_linear_probe(features_arr, labels_arr)
        min_count = min(np.bincount(labels_arr))
        n_splits = min(5, min_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, features_arr, labels_arr, cv=cv, scoring="accuracy")
            probe_acc = cv_scores.mean()
        else:
            clf.fit(features_arr, labels_arr)
            probe_acc = clf.score(features_arr, labels_arr)
        s_x = compute_s_x(probe_acc, num_classes)
    else:
        probe_acc = 0.0
        s_x = 0.0

    # Within-model R_QK gap
    rqk_gap = compute_within_model_rqk_gap(per_type_rqk)

    # M_AV range
    m_av_range = {
        "direct_mean": sum(per_type_mav.get("direct_evidence", [0])) / max(len(per_type_mav.get("direct_evidence", [1])), 1),
        "misleading_mean": sum(per_type_mav.get("misleading_hint", [0])) / max(len(per_type_mav.get("misleading_hint", [1])), 1),
    }

    # Scale-aware weights
    weights = compute_scale_aware_weights(rqk_gap, s_x, m_av_range, num_layers, num_heads)

    # Pathway analysis
    pathway = compute_pathway_switch_score(distributed_results) if distributed_results else {}

    remove_hooks(cache)

    # Summary
    print(f"  R_QK gap: {rqk_gap['gap']:.4f} (ratio={rqk_gap['ratio']:.1f}x)")
    print(f"  S_X: {s_x:.4f}")
    print(f"  M_AV direct: {m_av_range['direct_mean']:.4f}, misleading: {m_av_range['misleading_mean']:.4f}")
    print(f"  Scale-aware weights: α={weights['r_qk']:.3f} β={weights['m_av']:.3f} γ={weights['s_x']:.3f} δ={weights['c_do']:.3f}")
    if pathway:
        print(f"  Pathway: attn={pathway.get('attention_dominant_pct',0):.0f}% mlp={pathway.get('mlp_dominant_pct',0):.0f}%")
        print(f"  Attn frac={pathway.get('avg_attention_fraction',0):.3f} MLP frac={pathway.get('avg_mlp_fraction',0):.3f}")

    # Per-type ICI with scale-aware weights
    per_type_sa_ici = {}
    for rt in per_type_rqk:
        avg_rqk = sum(per_type_rqk[rt]) / len(per_type_rqk[rt])
        avg_mav = sum(per_type_mav[rt]) / len(per_type_mav[rt])
        per_type_sa_ici[rt] = round(compute_scale_aware_ici(avg_rqk, avg_mav, s_x, 0.0, weights), 4)

    return {
        "model_name": model_name,
        "arch_info": arch_info,
        "rqk_gap": rqk_gap,
        "S_X": round(s_x, 4),
        "probe_accuracy": round(float(probe_acc), 4),
        "m_av_range": m_av_range,
        "scale_aware_weights": weights,
        "per_type_rqk": {rt: round(sum(v)/len(v), 4) for rt, v in per_type_rqk.items()},
        "per_type_mav": {rt: round(sum(v)/len(v), 4) for rt, v in per_type_mav.items()},
        "per_type_ici_fixed": {rt: round(sum(v)/len(v), 4) for rt, v in per_type_ici_fixed.items()},
        "per_type_ici_calibrated": per_type_sa_ici,
        "pathway_analysis": pathway,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2,gpt2-medium,distilgpt2")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")

    model_names = [m.strip() for m in args.model.split(",")]
    results = []

    for name in model_names:
        model_id = MODELS.get(name, name)
        r = evaluate_model_v05(name, model_id, samples, device, args.limit)
        if r:
            results.append(r)

    # Calibrate across models
    calibrated = calibrate_weights_from_runs(results)

    # Print final comparison table
    print(f"\n{'='*90}")
    print("v0.5 SCALE-AWARE CALIBRATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':15s} {'Depth':>5s} {'R_QK gap':>10s} {'R_QK ratio':>10s} {'S_X':>8s} {'α':>6s} {'β':>6s} {'γ':>6s} {'δ':>6s} {'Pathway':>12s}")
    print(f"{'-'*85}")

    for r in results:
        name = r["model_name"]
        cal = calibrated.get(name, {})
        w = cal.get("weights", {})
        depth = r["arch_info"].get("num_layers", "?")
        pw = r.get("pathway_analysis", {})
        pw_str = ""
        if pw:
            pw_str = f"attn={pw.get('attention_dominant_pct',0):.0f}%"
        print(f"{name:15s} {str(depth):>5s} {r['rqk_gap']['gap']:10.4f} {r['rqk_gap']['ratio']:10.1f}x {r['S_X']:8.4f} {w.get('r_qk',0):6.4f} {w.get('m_av',0):6.4f} {w.get('s_x',0):6.4f} {w.get('c_do',0):6.4f} {pw_str:>12s}")

    # Per-type ICI comparison under fixed vs calibrated
    print(f"\n{'Type':20s} {'Fixed(distil)':>12s} {'Calib(distil)':>12s} {'Fixed(gpt2)':>12s} {'Calib(gpt2)':>12s} {'Fixed(med)':>12s} {'Calib(med)':>12s}")
    print(f"{'-'*85}")
    for rt in ["direct_evidence", "misleading_hint", "conflict", "evidence_gap", "multi_step"]:
        row = f"{rt:20s}"
        for name in ["distilgpt2", "gpt2", "gpt2-medium"]:
            model_r = next((r for r in results if r["model_name"] == name), None)
            if model_r:
                fixed = model_r["per_type_ici_fixed"].get(rt, 0)
                calib = model_r["per_type_ici_calibrated"].get(rt, 0)
                row += f" {fixed:12.4f} {calib:12.4f}"
        print(row)

    # Save results
    output_path = Path(__file__).parent.parent / "reports" / "v05_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"models": results, "calibration": calibrated}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
