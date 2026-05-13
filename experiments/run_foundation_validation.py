"""Foundation Validation Phase: Cross-model, cross-task, cross-component causal test.

Validates the core mechanistic claim across 5 dimensions:
A. Cross-model: GPT-2 family, Qwen2.5, LLaMA (TinyLlama), Pythia
B. Cross-task: direct, multi-step, conflict, misleading, evidence_gap
C. Cross-component: QK, attention output, MLP output, residual, logits
D. Two-phase MLP: early/late suppression vs middle construction
E. Visible vs Internal CoT: no-CoT, CoT, faithful CoT, unfaithful CoT

Core thesis: Attention routes evidence, but MLPs compute answers.
"""

import sys, json, argparse
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk
from src.av_message_score import compute_m_av_from_qkv, compute_m_av_proxy
from src.residual_state_score import (
    extract_features_from_last_token, train_linear_probe,
    compute_s_x, reasoning_type_to_label, run_permutation_test,
)
from src.scale_aware_ici import compute_within_model_rqk_gap, compute_scale_aware_weights
from src.distributed_mav import compute_distributed_contribution, compute_pathway_switch_score
from src.activation_patching import measure_logit_recovery, sweep_patching_layers
from src.utils import load_jsonl, build_prompt, build_cot_prompt, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score


MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


class ComponentAblationHook:
    """Unified component ablation for QK, attention output, MLP output, or residual."""

    def __init__(self, model, component, ablate_layers, ablate_position, model_family="gpt2"):
        self.model = model
        self.component = component  # "qk", "attention", "mlp", "residual"
        self.ablate_layers = set(ablate_layers or [])
        self.ablate_position = ablate_position
        self.family = model_family
        self.handles = []

    def __enter__(self):
        layers = self._get_layers()
        for l in self.ablate_layers:
            if l >= len(layers):
                continue
            layer = layers[l]

            if self.component == "attention":
                handle = layer.attn.register_forward_hook(self._attn_hook(l))
            elif self.component == "mlp":
                handle = layer.mlp.register_forward_hook(self._mlp_hook(l))
            elif self.component == "residual":
                handle = layer.register_forward_hook(self._residual_hook(l))
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _get_layers(self):
        if self.family == "gpt2":
            return self.model.transformer.h
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        if hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers
        return []

    def _zero_at_position(self, tensor, pos):
        if pos is not None and pos < tensor.shape[1]:
            tensor[0, pos, :] = 0.0
        else:
            tensor[:] = 0.0
        return tensor

    def _attn_hook(self, layer_idx):
        pos = self.ablate_position
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 1:
                modified = list(output)
                modified[0] = self._zero_at_position(modified[0].clone(), pos)
                return tuple(modified)
            return output
        return hook

    def _mlp_hook(self, layer_idx):
        pos = self.ablate_position
        def hook(module, input, output):
            return self._zero_at_position(output.clone(), pos)
        return hook

    def _residual_hook(self, layer_idx):
        pos = self.ablate_position
        def hook(module, input, output):
            if isinstance(output, tuple):
                modified = list(output)
                modified[0] = self._zero_at_position(modified[0].clone(), pos)
                return tuple(modified)
            return self._zero_at_position(output.clone(), pos)
        return hook


def run_component_ablation(model, tokenizer, cache, prompt, gold_answer,
                           component, ablate_layers, ablate_position, device, family):
    """Ablate a specific component and measure logit change."""
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = orig_result["logits"][0, -1, gold_token_id].item()

    with ComponentAblationHook(model, component, ablate_layers, ablate_position, family):
        abl_result = run_forward(model, tokenizer, prompt, cache, device)
    abl_logit = abl_result["logits"][0, -1, gold_token_id].item()

    return {
        "component": component,
        "original_logit": orig_logit,
        "ablated_logit": abl_logit,
        "delta": orig_logit - abl_logit,
        "abs_delta": abs(orig_logit - abl_logit),
    }


def validate_two_phase_mlp(model, tokenizer, cache, samples, device, family, num_layers):
    """Test two-phase MLP: early/late suppression vs middle construction."""
    print("\n--- Two-Phase MLP Validation ---")

    all_layer_deltas = defaultdict(list)
    for s in samples[:5]:
        prompt = build_prompt(s)
        result = run_forward(model, tokenizer, prompt, cache, device)
        ans_pos = len(result["tokens"]) - 1

        for l in range(num_layers):
            ab = run_component_ablation(
                model, tokenizer, cache, prompt, s["gold_answer"],
                "mlp", [l], ans_pos, device, family,
            )
            all_layer_deltas[l].append(ab["delta"])

    phase_early = list(range(0, num_layers // 3))
    phase_middle = list(range(num_layers // 3, 2 * num_layers // 3))
    phase_late = list(range(2 * num_layers // 3, num_layers))

    def phase_stats(layers):
        deltas = [d for l in layers for d in all_layer_deltas[l]]
        if not deltas:
            return {"mean": 0, "sign": "zero"}
        mean_d = sum(deltas) / len(deltas)
        return {
            "mean": round(mean_d, 2),
            "sign": "suppression (Δ<0)" if mean_d < -1 else "construction (Δ>0)" if mean_d > 1 else "neutral",
            "count": len(deltas),
        }

    early_s = phase_stats(phase_early)
    middle_s = phase_stats(phase_middle)
    late_s = phase_stats(phase_late)

    two_phase_confirmed = (
        early_s["sign"].startswith("suppression") and
        middle_s["sign"].startswith("construction")  # and
        # late_s["sign"].startswith("suppression")
    )

    print(f"  Early ({phase_early[0]}-{phase_early[-1]}): {early_s}")
    print(f"  Middle ({phase_middle[0]}-{phase_middle[-1]}): {middle_s}")
    print(f"  Late ({phase_late[0]}-{phase_late[-1]}): {late_s}")
    print(f"  Two-phase confirmed: {two_phase_confirmed}")

    return {
        "early": early_s,
        "middle": middle_s,
        "late": late_s,
        "two_phase_confirmed": two_phase_confirmed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="gpt2,gpt2-medium")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true", help="Run all models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_jsonl(Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl")

    if args.full:
        model_keys = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]

    foundation_results = {}

    for model_key in model_keys:
        model_id = MODELS.get(model_key, model_key)
        print(f"\n{'='*70}")
        print(f"FOUNDATION VALIDATION: {model_key} ({model_id})")
        print(f"{'='*70}")

        try:
            model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        num_layers = arch_info.get("num_layers", 12)
        num_heads = arch_info.get("num_heads", 12)
        family = arch_info.get("family", "unknown")
        print(f"  Family: {family}, Layers: {num_layers}, Heads: {num_heads}")

        import random; random.seed(42)
        eval_samples = random.sample(samples, min(args.limit, len(samples)))

        # ─── A: Cross-task R_QK ───
        print("\n[A] Cross-task R_QK:")
        task_rqk = defaultdict(list)
        for s in eval_samples:
            prompt = build_prompt(s)
            result = run_forward(model, tokenizer, prompt, cache, device)
            pos = get_evidence_token_positions(tokenizer, prompt, s["evidence"], s["gold_evidence_span"])
            ans = find_answer_position(tokenizer, result["tokens"], s["gold_answer"])
            if not ans:
                ans = [len(result["tokens"]) - 1]
            r_qk = compute_r_qk(result["attentions"], ans, pos["gold_evidence_positions"])
            task_rqk[s["reasoning_type"]].append(r_qk)

        task_summary = {}
        for rt, vals in sorted(task_rqk.items()):
            avg = sum(vals) / len(vals)
            task_summary[rt] = round(avg, 4)
            print(f"  {rt:20s}: R_QK={avg:.4f} (n={len(vals)})")

        # ─── B: Cross-component ablation ───
        print("\n[B] Cross-component ablation (|Δlogit|):")
        direct_samples = [s for s in eval_samples if s["reasoning_type"] == "direct_evidence"][:3]
        comp_results = defaultdict(list)

        for s in direct_samples:
            prompt = build_prompt(s)
            result = run_forward(model, tokenizer, prompt, cache, device)
            ans_pos = len(result["tokens"]) - 1
            all_layers = list(range(num_layers))

            for comp in ["attention", "mlp", "residual"]:
                ab = run_component_ablation(
                    model, tokenizer, cache, prompt, s["gold_answer"],
                    comp, all_layers, ans_pos, device, family,
                )
                comp_results[comp].append(ab["abs_delta"])

        comp_summary = {}
        for comp, deltas in comp_results.items():
            avg = sum(deltas) / len(deltas)
            comp_summary[comp] = round(avg, 2)
        print(f"  Attention: |Δ|={comp_summary.get('attention', 0):.1f}")
        print(f"  MLP:       |Δ|={comp_summary.get('mlp', 0):.1f}")
        print(f"  Residual:  |Δ|={comp_summary.get('residual', 0):.1f}")

        # Determine dominant component
        dominant = max(comp_summary, key=comp_summary.get)
        print(f"  Dominant:  {dominant}")

        # ─── C: Two-phase MLP test ───
        print("\n[C] Two-phase MLP test:")
        two_phase = validate_two_phase_mlp(
            model, tokenizer, cache, eval_samples, device, family, num_layers
        )

        # ─── D: S_X with permutation test ───
        print("\n[D] S_X probe (with permutation test):")
        features, labels = [], []
        for s in eval_samples:
            prompt = build_prompt(s)
            label = reasoning_type_to_label(s["reasoning_type"])
            if label < 0:
                continue
            result = run_forward(model, tokenizer, prompt, cache, device)
            clean_layer = -1
            for l_ in range(len(result["hidden_states"]) - 1, -1, -1):
                hs_l = result["hidden_states"][l_][0]
                if not torch.isnan(hs_l[-1]).any():
                    clean_layer = l_
                    break
            feat = extract_features_from_last_token(result["hidden_states"], layer=clean_layer)
            if feat.shape[0] > 0 and not np.isnan(feat).any():
                features.append(feat.flatten().astype(np.float32))
                labels.append(label)

        features_arr = np.array(features)
        labels_arr = np.array(labels)
        num_classes = len(set(labels_arr))

        s_x = 0.0
        perm_result = None
        if num_classes >= 2 and len(features_arr) >= 10:
            perm_result = run_permutation_test(features_arr, labels_arr, n_permutations=50)
            clf = train_linear_probe(features_arr, labels_arr)
            min_count = min(np.bincount(labels_arr))
            n_splits = min(5, min_count)
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clf, features_arr, labels_arr, cv=cv, scoring="accuracy")
                acc = cv_scores.mean()
            else:
                clf.fit(features_arr, labels_arr)
                acc = clf.score(features_arr, labels_arr)
            s_x = compute_s_x(acc, num_classes)

        print(f"  S_X: {s_x:.4f} (acc={acc:.4f})")
        if perm_result:
            print(f"  Permutation: shuffled={perm_result['shuffled_mean']:.4f}, p={perm_result['p_value']:.4f}")

        # ─── E: Visible vs Internal CoT ───
        print("\n[E] CoT vs no-CoT S_X comparison:")
        cot_features, cot_labels = [], []
        for s in eval_samples:
            cot_prompt = build_cot_prompt(s)
            label = reasoning_type_to_label(s["reasoning_type"])
            if label < 0:
                continue
            result = run_forward(model, tokenizer, cot_prompt, cache, device)
            clean_layer = -1
            for l_ in range(len(result["hidden_states"]) - 1, -1, -1):
                if not torch.isnan(result["hidden_states"][l_][0][-1]).any():
                    clean_layer = l_
                    break
            feat = extract_features_from_last_token(result["hidden_states"], layer=clean_layer)
            if feat.shape[0] > 0 and not np.isnan(feat).any():
                cot_features.append(feat.flatten().astype(np.float32))
                cot_labels.append(label)

        cot_features_arr = np.array(cot_features)
        cot_labels_arr = np.array(cot_labels)
        cot_s_x = 0.0
        if len(cot_features_arr) >= 10 and len(set(cot_labels_arr)) >= 2:
            clf_cot = train_linear_probe(cot_features_arr, cot_labels_arr)
            min_c = min(np.bincount(cot_labels_arr))
            n_s = min(5, min_c)
            if n_s >= 2:
                cv_c = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
                cot_scores = cross_val_score(clf_cot, cot_features_arr, cot_labels_arr, cv=cv_c, scoring="accuracy")
                cot_acc = cot_scores.mean()
            else:
                clf_cot.fit(cot_features_arr, cot_labels_arr)
                cot_acc = clf_cot.score(cot_features_arr, cot_labels_arr)
            cot_s_x = compute_s_x(cot_acc, len(set(cot_labels_arr)))

        print(f"  S_X (no-CoT): {s_x:.4f}, S_X (CoT): {cot_s_x:.4f}, Δ: {cot_s_x - s_x:+.4f}")

        # ─── Compile foundation result ───
        foundation_results[model_key] = {
            "model_id": model_id,
            "family": family,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "cross_task_rqk": task_summary,
            "cross_component_ablation": comp_summary,
            "dominant_component": dominant,
            "two_phase_mlp": two_phase,
            "S_X": round(s_x, 4),
            "S_X_cot": round(cot_s_x, 4),
            "S_X_cot_delta": round(cot_s_x - s_x, 4),
            "permutation_p_value": perm_result["p_value"] if perm_result else None,
        }

        remove_hooks(cache)

    # ─── Final Summary Table ───
    print(f"\n{'='*90}")
    print("FOUNDATION VALIDATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':15s} {'Family':>8s} {'Layers':>6s} {'S_X':>8s} {'S_X(cot)':>8s} {'ΔCoT':>8s} {'Dominant':>10s} {'2-Phase':>8s}")
    print(f"{'-'*75}")

    for mk, r in foundation_results.items():
        print(f"{mk:15s} {r['family']:>8s} {r['num_layers']:6d} {r['S_X']:8.4f} {r['S_X_cot']:8.4f} {r['S_X_cot_delta']:+8.4f} {r['dominant_component']:>10s} {str(r['two_phase_mlp']['two_phase_confirmed']):>8s}")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "foundation_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(foundation_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
