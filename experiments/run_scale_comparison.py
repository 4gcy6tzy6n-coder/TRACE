"""v0.4: Multi-model ICI comparison across model scales.

Tests whether ICI patterns hold across different model sizes and architectures.
Primary question: Does ICI increase with model scale for reasoning-heavy types?
Secondary: Do larger models show faithful > unfaithful CoT discrimination?

Usage:
    python experiments/run_scale_comparison.py --model gpt2
    python experiments/run_scale_comparison.py --model gpt2-medium
    python experiments/run_scale_comparison.py --model all  # compare all available models
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk, compute_r_qk_per_head, identify_evidence_routing_heads


def filter_nan_attentions(attentions):
    """Remove layers with NaN attention weights (float16 overflow issue)."""
    clean = []
    nan_layers = []
    for l, attn in enumerate(attentions):
        if not torch.isnan(attn).any():
            clean.append(attn)
        else:
            nan_layers.append(l)
    if nan_layers:
        print(f"  [NaN layers filtered: {nan_layers}]")
    return clean if clean else attentions
from src.av_message_score import compute_m_av_from_qkv, compute_m_av_proxy
from src.residual_state_score import (
    extract_features_from_last_token, train_linear_probe,
    compute_s_x, reasoning_type_to_label,
)
from src.ici_calculator import compute_ici_for_sample
from src.utils import load_jsonl, build_prompt, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score


AVAILABLE_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "distilgpt2": "distilgpt2",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
}


def evaluate_model(model_name: str, model_id: str, samples, device: str, limit: int = 0,
                   compute_proxy_mav: bool = True, compute_true_mav: bool = True):
    """Run ICI evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_id})")
    print(f"{'='*60}")

    try:
        model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
    except Exception as e:
        print(f"  SKIP: Failed to load {model_id}: {e}")
        return None

    if limit > 0:
        import random
        random.seed(42)
        eval_samples = random.sample(samples, min(limit, len(samples)))
    else:
        eval_samples = samples

    print(f"  Evaluating {len(eval_samples)} samples")
    print(f"  Architecture: {arch_info['arch']}")
    print(f"  Layers: {arch_info.get('num_layers', '?')}, Heads: {arch_info.get('num_heads', '?')}")

    # Collect hidden states for S_X probe
    features = []
    labels = []

    for s in eval_samples:
        prompt = build_prompt(s)
        label = reasoning_type_to_label(s["reasoning_type"])
        if label < 0:
            continue
        result = run_forward(model, tokenizer, prompt, cache, device)
        # Find deepest clean layer (no NaN in last token)
        clean_layer = -1
        for l in range(len(result["hidden_states"]) - 1, -1, -1):
            hs = result["hidden_states"][l][0]  # [seq, hidden]
            if not torch.isnan(hs[-1]).any():
                clean_layer = l
                break
        feat = extract_features_from_last_token(result["hidden_states"], layer=clean_layer)
        if feat.shape[0] > 0 and not np.isnan(feat).any():
            features.append(feat.flatten().astype(np.float32))
            labels.append(label)

    features = np.array(features)
    labels_arr = np.array(labels)
    num_classes = len(set(labels_arr))

    # S_X probe
    if num_classes >= 2 and len(features) >= 10:
        clf = train_linear_probe(features, labels_arr)
        min_count = min(np.bincount(labels_arr))
        n_splits = min(5, min_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, features, labels_arr, cv=cv, scoring="accuracy")
            probe_acc = cv_scores.mean()
        else:
            clf.fit(features, labels_arr)
            probe_acc = clf.score(features, labels_arr)
        s_x = compute_s_x(probe_acc, num_classes)
    else:
        probe_acc = 0.0
        s_x = 0.0

    print(f"  S_X probe: acc={probe_acc:.4f}, S_X={s_x:.4f}")

    # Per-sample ICI (subset for speed)
    ici_samples = eval_samples[:min(20, len(eval_samples))]
    results = []

    for sample in ici_samples:
        prompt = build_prompt(sample)
        result = run_forward(model, tokenizer, prompt, cache, device)

        pos_info = get_evidence_token_positions(
            tokenizer, prompt, sample["evidence"], sample["gold_evidence_span"]
        )
        ev_pos = pos_info["gold_evidence_positions"]
        ans_positions = find_answer_position(tokenizer, result["tokens"], sample["gold_answer"])
        if not ans_positions:
            ans_positions = [len(result["tokens"]) - 1]

        clean_attns = filter_nan_attentions(result["attentions"])
        r_qk = compute_r_qk(clean_attns, ans_positions, ev_pos)

        m_av = 0.0
        if compute_true_mav and result["q_per_layer"]:
            ans_token_id = tokenizer.encode(sample["gold_answer"], add_special_tokens=False)
            if ans_token_id:
                try:
                    m_av_result = compute_m_av_from_qkv(
                        model, result["q_per_layer"], result["k_per_layer"],
                        result["v_per_layer"], ev_pos,
                        ans_positions[0], ans_token_id[0],
                        num_heads=arch_info.get("num_heads", 12),
                        head_dim=arch_info.get("head_dim", 64),
                    )
                    m_av = m_av_result["M_AV"]
                except Exception:
                    m_av = 0.0

        if compute_proxy_mav and m_av == 0.0:
            try:
                proxy = compute_m_av_proxy(
                    model, tokenizer, cache, prompt,
                    sample["gold_evidence_span"], sample["gold_answer"], device
                )
                m_av = proxy["M_AV"]
            except Exception:
                m_av = 0.0

        ici = compute_ici_for_sample(sample["id"], r_qk, m_av, s_x, 0.0)
        ici["reasoning_type"] = sample["reasoning_type"]
        results.append(ici)

    # Per-type summary
    type_summary = {}
    for r in results:
        rt = r["reasoning_type"]
        if rt not in type_summary:
            type_summary[rt] = {"ici": [], "rqk": [], "mav": []}
        type_summary[rt]["ici"].append(r["ICI"])
        type_summary[rt]["rqk"].append(r["R_QK"])
        type_summary[rt]["mav"].append(r["M_AV"])

    print(f"\n  {'Type':20s} {'n':>3s} {'ICI':>8s} {'R_QK':>8s} {'M_AV':>8s}")
    print(f"  {'-'*50}")
    for rt, vals in sorted(type_summary.items()):
        avg_ici = sum(vals["ici"]) / len(vals["ici"])
        avg_rqk = sum(vals["rqk"]) / len(vals["rqk"])
        avg_mav = sum(vals["mav"]) / len(vals["mav"])
        print(f"  {rt:20s} {len(vals['ici']):3d} {avg_ici:8.4f} {avg_rqk:8.4f} {avg_mav:8.4f}")

    remove_hooks(cache)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "arch_info": arch_info,
        "S_X": round(s_x, 4),
        "probe_accuracy": round(float(probe_acc), 4),
        "num_samples_evaluated": len(ici_samples),
        "per_type": {rt: {k: round(sum(v)/len(v), 4) for k, v in vals.items()}
                     for rt, vals in type_summary.items()},
        "overall_avg_ICI": round(sum(r["ICI"] for r in results) / len(results), 4) if results else 0,
        "overall_avg_R_QK": round(sum(r["R_QK"] for r in results) / len(results), 4) if results else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Scale comparison across models")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model key or 'all' for all available models")
    parser.add_argument("--limit", type=int, default=50, help="Samples to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)
    print(f"Dataset: {len(samples)} samples")

    if args.model == "all":
        models_to_run = list(AVAILABLE_MODELS.items())
    else:
        if args.model in AVAILABLE_MODELS:
            models_to_run = [(args.model, AVAILABLE_MODELS[args.model])]
        else:
            models_to_run = [(args.model, args.model)]  # try as-is HF ID

    all_results = []
    for model_name, model_id in models_to_run:
        result = evaluate_model(model_name, model_id, samples, device, limit=args.limit)
        if result:
            all_results.append(result)

    # Comparison table
    print(f"\n{'='*80}")
    print("SCALE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':20s} {'Layers':>7s} {'Heads':>7s} {'S_X':>8s} {'ICI':>8s}")
    print(f"{'-'*55}")
    for r in all_results:
        layers = r["arch_info"].get("num_layers", "?")
        heads = r["arch_info"].get("num_heads", "?")
        print(f"{r['model_name']:20s} {str(layers):>7s} {str(heads):>7s} {r['S_X']:8.4f} {r['overall_avg_ICI']:8.4f}")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "scale_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
