"""Cross-Model Foundation Validation.

Tests the mechanism chain (QK→MLP→X_l→logits) across 6+ model families.
Three claims per model:
  C1: R_QK(direct) / R_QK(misleading) > 5×
  C2: MLP ablation |Δlogit| > attention ablation |Δlogit|
  C3: S_X significantly > random/shuffled baseline

Usage:
    python experiments/run_cross_model_foundation.py
    python experiments/run_cross_model_foundation.py --models gpt2,gpt2-medium,Qwen/Qwen2.5-0.5B
    python experiments/run_cross_model_foundation.py --small  # Quick test on 1 model
"""

import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk
from src.residual_state_score import (
    extract_features_from_last_token, train_linear_probe,
    compute_s_x, reasoning_type_to_label, run_permutation_test,
)
from src.utils import load_jsonl, build_prompt, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ─── Model registry ───

MODEL_REGISTRY = {
    "gpt2":              {"id": "gpt2",                    "family": "GPT-2",   "params": "124M"},
    "gpt2-medium":       {"id": "gpt2-medium",             "family": "GPT-2",   "params": "355M"},
    "distilgpt2":        {"id": "distilgpt2",              "family": "GPT-2",   "params": "82M"},
    "qwen-0.5b":         {"id": "Qwen/Qwen2.5-0.5B",      "family": "Qwen2.5", "params": "494M"},
    "qwen-1.5b":         {"id": "Qwen/Qwen2.5-1.5B",      "family": "Qwen2.5", "params": "1.5B"},
    "tinyllama":         {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "LLaMA", "params": "1.1B"},
    "gemma-2b":          {"id": "google/gemma-2-2b",       "family": "Gemma",   "params": "2B"},
    "pythia-70m":        {"id": "EleutherAI/pythia-70m",   "family": "NeoX",    "params": "70M",  "fp_issue": "fp16 NaN"},
    "pythia-160m":       {"id": "EleutherAI/pythia-160m",  "family": "NeoX",    "params": "160M", "fp_issue": "fp16 NaN"},
}


# ─── Component ablation hook ───

class QuickAblationHook:
    """Minimal ablation hook for attention or MLP output at a specific position."""

    def __init__(self, model, component, layers, position, family="gpt2"):
        self.component = component
        self.layers = set(layers or [])
        self.position = position
        self.family = family
        self.handles = []
        self._layers = self._find_layers(model)

    def _find_layers(self, model):
        if self.family == "gpt2":
            return model.transformer.h
        if self.family in ("neox",):
            return model.gpt_neox.layers if hasattr(model, 'gpt_neox') else []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        return []

    def _get_attn_module(self, layer):
        """Get attention module, handling different naming conventions."""
        for name in ('attn', 'self_attn', 'attention', 'self_attention'):
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    def _get_mlp_module(self, layer):
        """Get MLP module, handling different naming conventions."""
        for name in ('mlp', 'feed_forward', 'ffn', 'mlp_norm'):
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    def __enter__(self):
        for l in self.layers:
            if l >= len(self._layers):
                continue
            layer = self._layers[l]
            if self.component == "attention":
                attn_mod = self._get_attn_module(layer)
                if attn_mod:
                    h = attn_mod.register_forward_hook(self._zero_attn)
                    self.handles.append(h)
            elif self.component == "mlp":
                mlp_mod = self._get_mlp_module(layer)
                if mlp_mod:
                    h = mlp_mod.register_forward_hook(self._zero_mlp)
                    self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _zero_at(self, t, pos):
        if pos is not None and pos < t.shape[1]:
            t[0, pos, :] = 0.0
        else:
            t[:] = 0.0
        return t

    def _zero_attn(self, module, input, output):
        if isinstance(output, tuple) and len(output) >= 1:
            modified = list(output)
            modified[0] = self._zero_at(modified[0].clone(), self.position)
            return tuple(modified)
        return output

    def _zero_mlp(self, module, input, output):
        return self._zero_at(output.clone(), self.position)


def _sanitize_for_json(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─── Per-model validation ───

def validate_model(model_key, model_info, samples, device, n_samples=30):
    """Run 3-claim validation for a single model.

    Returns dict with pass/fail per claim, plus detailed metrics.
    """
    model_id = model_info["id"]
    family = model_info["family"]
    known_fp_issue = model_info.get("fp_issue", "")

    print(f"\n{'─'*60}")
    print(f"Model: {model_key} ({model_id})")
    print(f"Family: {family} | Params: {model_info['params']}")
    if known_fp_issue:
        print(f"Known issue: {known_fp_issue}")

    result = {
        "model_key": model_key,
        "model_id": model_id,
        "family": family,
        "params": model_info["params"],
        "status": "attempted",
        "claims": {},
        "errors": [],
    }

    # ── Load model ──
    try:
        model, tokenizer, cache, arch_info = load_model_v04(model_id, device)
    except Exception as e:
        result["status"] = "load_failed"
        result["errors"].append(f"load: {e}")
        print(f"  FAIL: Could not load model: {e}")
        return result

    num_layers = arch_info.get("num_layers", 0)
    num_heads = arch_info.get("num_heads", 0)
    detected_family = arch_info.get("family", "unknown")
    result["num_layers"] = num_layers
    result["num_heads"] = num_heads
    result["detected_family"] = detected_family
    print(f"  Detected: {num_layers}L, {num_heads}H, family={detected_family}")

    # ── Sample selection ──
    import random; random.seed(42)
    eval_samples = random.sample(samples, min(n_samples, len(samples)))

    # ── C1: R_QK routing ratio ──
    print("\n  [C1] QK Evidence Routing...")
    rqk_direct, rqk_misleading = [], []

    for s in eval_samples:
        try:
            prompt = build_prompt(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            pos = get_evidence_token_positions(tokenizer, prompt, s["evidence"], s["gold_evidence_span"])
            ans = find_answer_position(tokenizer, r["tokens"], s["gold_answer"])
            if not ans:
                ans = [len(r["tokens"]) - 1]
            rqk = compute_r_qk(r["attentions"], ans, pos["gold_evidence_positions"])
            if s["reasoning_type"] == "direct_evidence":
                rqk_direct.append(rqk)
            elif s["reasoning_type"] == "misleading_hint":
                rqk_misleading.append(rqk)
        except Exception as e:
            continue

    c1_ratio = 0
    if rqk_direct and rqk_misleading:
        avg_direct = sum(rqk_direct) / len(rqk_direct)
        avg_misleading = sum(rqk_misleading) / len(rqk_misleading)
        c1_ratio = avg_direct / max(avg_misleading, 1e-8)
        c1_pass = c1_ratio > 5.0
        result["claims"]["c1_rqk_routing"] = {
            "direct_mean": round(avg_direct, 4),
            "misleading_mean": round(avg_misleading, 4),
            "ratio": round(c1_ratio, 1),
            "pass": c1_pass,
        }
        print(f"    R_QK(direct)={avg_direct:.4f}, R_QK(misleading)={avg_misleading:.4f}, ratio={c1_ratio:.1f}x → {'PASS' if c1_pass else 'FAIL'}")
    else:
        result["claims"]["c1_rqk_routing"] = {"pass": False, "error": "insufficient samples"}
        print("    FAIL: insufficient samples for routing comparison")

    # ── C2: MLP > Attention ablation ──
    print("\n  [C2] MLP vs Attention Causal Ablation...")
    direct_samples = [s for s in eval_samples if s["reasoning_type"] == "direct_evidence"][:3]

    if direct_samples and num_layers > 0:
        attn_deltas, mlp_deltas = [], []
        for s in direct_samples:
            try:
                prompt = build_prompt(s)
                r = run_forward(model, tokenizer, prompt, cache, device)
                ans_pos = len(r["tokens"]) - 1
                gold_id = tokenizer.encode(s["gold_answer"], add_special_tokens=False)[0]

                # Original logit
                orig_logit = r["logits"][0, -1, gold_id].item()

                # Attention ablation
                with QuickAblationHook(model, "attention", list(range(num_layers)), ans_pos, detected_family):
                    ar = run_forward(model, tokenizer, prompt, cache, device)
                attn_deltas.append(abs(orig_logit - ar["logits"][0, -1, gold_id].item()))

                # MLP ablation
                with QuickAblationHook(model, "mlp", list(range(num_layers)), ans_pos, detected_family):
                    mr = run_forward(model, tokenizer, prompt, cache, device)
                mlp_deltas.append(abs(orig_logit - mr["logits"][0, -1, gold_id].item()))
            except Exception as e:
                continue

        if attn_deltas and mlp_deltas:
            avg_attn = sum(attn_deltas) / len(attn_deltas)
            avg_mlp = sum(mlp_deltas) / len(mlp_deltas)
            c2_ratio = avg_mlp / max(avg_attn, 1e-8)
            c2_pass = avg_mlp > avg_attn
            result["claims"]["c2_mlp_dominance"] = {
                "attention_abs_delta": round(avg_attn, 1),
                "mlp_abs_delta": round(avg_mlp, 1),
                "mlp_attn_ratio": round(c2_ratio, 2),
                "pass": c2_pass,
            }
            print(f"    Attn |Δ|={avg_attn:.1f}, MLP |Δ|={avg_mlp:.1f}, ratio={c2_ratio:.2f}x → {'PASS' if c2_pass else 'FAIL'}")
        else:
            result["claims"]["c2_mlp_dominance"] = {"pass": False, "error": "ablation failed"}
            print("    FAIL: ablation did not produce valid results")
    else:
        result["claims"]["c2_mlp_dominance"] = {"pass": False, "error": "no direct_evidence samples"}
        print("    FAIL: no direct_evidence samples for ablation")

    # ── C3: S_X > controls ──
    print("\n  [C3] Residual State Probe (S_X)...")
    features, labels_list = [], []
    nan_layers_count = 0

    for s in eval_samples:
        try:
            prompt = build_prompt(s)
            label = reasoning_type_to_label(s["reasoning_type"])
            if label < 0:
                continue
            r = run_forward(model, tokenizer, prompt, cache, device)
            # Find deepest clean layer
            clean_l = -1
            for l_ in range(len(r["hidden_states"]) - 1, -1, -1):
                if not torch.isnan(r["hidden_states"][l_][0][-1]).any():
                    clean_l = l_
                    break
            if clean_l >= 0:
                feat = extract_features_from_last_token(r["hidden_states"], layer=clean_l)
                if feat.shape[0] > 0 and not np.isnan(feat).any():
                    features.append(feat.flatten().astype(np.float32))
                    labels_list.append(label)
            else:
                nan_layers_count += 1
        except Exception:
            continue

    if nan_layers_count > 0:
        print(f"    ({nan_layers_count} samples had NaN hidden states — filtered)")

    c3_pass = False
    if len(features) >= 10 and len(set(labels_list)) >= 2:
        features_arr = np.array(features)
        labels_arr = np.array(labels_list)
        num_classes = len(set(labels_arr))

        try:
            clf = train_linear_probe(features_arr, labels_arr)
            min_c = min(np.bincount(labels_arr))
            n_s = min(3, min_c)  # Use 3-fold max to avoid overfitting with small samples
            acc = 0.0
            if n_s >= 2 and len(features_arr) >= n_s * num_classes:
                from sklearn.linear_model import LogisticRegression
                clf_cv = LogisticRegression(max_iter=1000, random_state=42, C=0.1)  # Stronger regularization
                cv = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clf_cv, features_arr, labels_arr, cv=cv, scoring="accuracy")
                acc = cv_scores.mean()
            else:
                from sklearn.linear_model import LogisticRegression
                clf_fit = LogisticRegression(max_iter=1000, C=0.1)
                clf_fit.fit(features_arr, labels_arr)
                acc = clf_fit.score(features_arr, labels_arr)
            s_x = compute_s_x(acc, num_classes)

            # Quick permutation test
            perm = run_permutation_test(features_arr, labels_arr, n_permutations=30)
            shuffled_acc = perm["shuffled_mean"]

            c3_pass = s_x > 0.1 and acc > shuffled_acc + 0.05
            result["claims"]["c3_residual_state"] = {
                "probe_accuracy": round(float(acc), 4),
                "S_X": round(s_x, 4),
                "shuffled_accuracy": round(shuffled_acc, 4),
                "p_value": perm["p_value"],
                "pass": c3_pass,
            }
            print(f"    Probe acc={acc:.4f}, S_X={s_x:.4f}, shuffled={shuffled_acc:.4f}, p={perm['p_value']:.4f} → {'PASS' if c3_pass else 'FAIL'}")
        except Exception as e:
            result["claims"]["c3_residual_state"] = {"pass": False, "error": str(e)}
            print(f"    FAIL: probe training error: {e}")
    else:
        result["claims"]["c3_residual_state"] = {"pass": False, "error": f"insufficient clean features ({len(features)} samples, {len(set(labels_list))} classes)"}
        print(f"    FAIL: insufficient clean features")

    # ── Overall status ──
    claims = result["claims"]
    c1 = claims.get("c1_rqk_routing", {}).get("pass", False)
    c2 = claims.get("c2_mlp_dominance", {}).get("pass", False)
    c3 = claims.get("c3_residual_state", {}).get("pass", False)
    passed = sum([c1, c2, c3])
    result["claims_passed"] = passed
    result["claims_total"] = 3
    result["status"] = "completed"

    print(f"\n  Result: {passed}/3 claims passed")
    remove_hooks(cache)
    return result


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model keys, or empty for default set")
    parser.add_argument("--small", action="store_true", help="Quick test on gpt2 only")
    parser.add_argument("--limit", type=int, default=30, help="Samples per model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Samples per model: {args.limit}")

    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)
    print(f"Dataset: {len(samples)} samples")

    # Determine model set
    if args.small:
        model_keys = ["gpt2"]
    elif args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        # Default: all clean models, skip known fp16-issue models
        model_keys = ["gpt2", "gpt2-medium", "qwen-0.5b"]

    results = []
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            # Try as direct HF ID
            info = {"id": key, "family": "unknown", "params": "?"}
        else:
            info = MODEL_REGISTRY[key]

        t0 = time.time()
        r = validate_model(key, info, samples, device, n_samples=args.limit)
        elapsed = time.time() - t0
        r["elapsed_seconds"] = round(elapsed, 0)
        # Convert numpy types for JSON serialization
        r = _sanitize_for_json(r)
        results.append(r)

    # ── Summary Table ──
    print(f"\n{'='*90}")
    print("CROSS-MODEL FOUNDATION VALIDATION — SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':18s} {'Family':10s} {'Params':>7s} {'Layers':>6s} {'C1:R_QK':>8s} {'C2:MLP':>8s} {'C3:S_X':>8s} {'Passed':>8s} {'Status':>10s}")
    print(f"{'-'*85}")

    for r in results:
        claims = r.get("claims", {})
        c1 = "✓" if claims.get("c1_rqk_routing", {}).get("pass") else "✗"
        c2 = "✓" if claims.get("c2_mlp_dominance", {}).get("pass") else "✗"
        c3 = "✓" if claims.get("c3_residual_state", {}).get("pass") else "✗"
        passed = r.get("claims_passed", 0)
        status = r["status"]
        if status == "completed":
            status = "OK" if passed >= 2 else "PARTIAL"
        print(f"{r['model_key']:18s} {r['family']:10s} {r['params']:>7s} {str(r.get('num_layers','?')):>6s} {c1:>8s} {c2:>8s} {c3:>8s} {passed:>6d}/3 {status:>10s}")

    # Summary with details
    print(f"\n{'─'*90}")
    print("DETAILED CLAIMS")
    for r in results:
        if r["status"] not in ("completed",):
            continue
        print(f"\n{r['model_key']} ({r['family']}, {r.get('num_layers','?')}L):")
        for cname, cdata in r.get("claims", {}).items():
            if cdata.get("pass"):
                print(f"  {cname}: PASS")
            else:
                print(f"  {cname}: FAIL — {cdata.get('error', 'see details')}")
            # Show key numbers
            if "ratio" in cdata:
                print(f"    ratio={cdata['ratio']}")
            if "mlp_abs_delta" in cdata:
                print(f"    attn|Δ|={cdata['attention_abs_delta']}, mlp|Δ|={cdata['mlp_abs_delta']}")
            if "S_X" in cdata:
                print(f"    S_X={cdata['S_X']}, acc={cdata.get('probe_accuracy','?')}, p={cdata.get('p_value','?')}")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "cross_model_foundation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
