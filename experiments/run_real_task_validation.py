"""Real-task mechanism validation.

Tests whether the mechanism chain (QK→MLP→X_l→logits) holds on
real QA tasks, not just controlled synthetic samples.

Currently validates on the controlled dataset as baseline.
To add real datasets, provide paths to FEVER/HotpotQA JSON/JSONL files.

Usage:
    python experiments/run_real_task_validation.py
    python experiments/run_real_task_validation.py --fever data/fever.jsonl
    python experiments/run_real_task_validation.py --hotpotqa data/hotpotqa.json
"""

import sys, json, argparse
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
from src.real_task_loader import load_controlled_real_mix
from src.utils import build_prompt, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score


class QuickAblationHook:
    """See run_cross_model_foundation.py for full implementation."""
    def __init__(self, model, component, layers, position, family="gpt2"):
        self.component = component
        self.layers = set(layers or [])
        self.position = position
        self.family = family
        self.handles = []
        self._layers = self._find_layers(model)

    def _find_layers(self, model):
        for attr in ['transformer.h', 'gpt_neox.layers', 'model.layers']:
            obj = model
            for part in attr.split('.'):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                return obj
        return []

    def _get_module(self, layer, names):
        for n in names:
            if hasattr(layer, n):
                return getattr(layer, n)
        return None

    def __enter__(self):
        for l in self.layers:
            if l >= len(self._layers):
                continue
            layer = self._layers[l]
            if self.component == "attention":
                mod = self._get_module(layer, ('attn', 'self_attn', 'attention'))
            elif self.component == "mlp":
                mod = self._get_module(layer, ('mlp', 'feed_forward', 'ffn'))
            else:
                continue
            if mod:
                h = mod.register_forward_hook(self._make_hook())
                self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self):
        pos = self.position
        def hook(module, input, output):
            if self.component == "attention" and isinstance(output, tuple) and len(output) >= 1:
                modified = list(output)
                if pos is not None and pos < modified[0].shape[1]:
                    modified[0][0, pos, :] = 0.0
                else:
                    modified[0][:] = 0.0
                return tuple(modified)
            elif pos is not None and pos < output.shape[1]:
                out = output.clone()
                out[0, pos, :] = 0.0
                return out
            else:
                return torch.zeros_like(output)
        return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever", type=str, default="", help="Path to FEVER JSONL")
    parser.add_argument("--hotpotqa", type=str, default="", help="Path to HotpotQA JSON")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Real-Task Mechanism Validation — {args.model}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    real_paths = {}
    if args.fever:
        real_paths["fever"] = args.fever
    if args.hotpotqa:
        real_paths["hotpotqa"] = args.hotpotqa

    samples = load_controlled_real_mix(data_path, real_paths if real_paths else None,
                                        max_controlled=args.limit, max_real=30)

    # Count by source
    source_counts = defaultdict(int)
    for s in samples:
        source_counts[s.get("source", "controlled")] += 1
    print(f"Samples: {dict(source_counts)}")

    # Load model
    model, tokenizer, cache, arch_info = load_model_v04(args.model, device)
    num_layers = arch_info.get("num_layers", 12)
    family = arch_info.get("family", "unknown")
    print(f"Model: {arch_info['arch']} ({num_layers}L)")

    # ─── 1. R_QK by reasoning type ───
    print("\n─── R_QK: Evidence Routing by Type ───")
    rqk_by_type = defaultdict(list)
    for s in samples:
        prompt = build_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)
        pos = get_evidence_token_positions(tokenizer, prompt, s["evidence"], s["gold_evidence_span"])
        ans = find_answer_position(tokenizer, r["tokens"], s["gold_answer"])
        if not ans:
            ans = [len(r["tokens"]) - 1]
        rqk = compute_r_qk(r["attentions"], ans, pos["gold_evidence_positions"])
        rt = s.get("reasoning_type", "unknown")
        src = s.get("source", "controlled")
        rqk_by_type[f"{rt} ({src})"].append(rqk)

    for label, vals in sorted(rqk_by_type.items()):
        print(f"  {label:30s}: R_QK={sum(vals)/len(vals):.4f} (n={len(vals)})")

    # ─── 2. MLP vs Attention ablation ───
    print("\n─── MLP vs Attention Ablation ───")
    direct_samples = [s for s in samples if s.get("reasoning_type") in ("direct_evidence", "multi_step")][:3]

    if direct_samples:
        for s in direct_samples:
            prompt = build_prompt(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            ans_pos = len(r["tokens"]) - 1
            try:
                gold_id = tokenizer.encode(s["gold_answer"], add_special_tokens=False)[0]
            except Exception:
                continue
            orig_logit = r["logits"][0, -1, gold_id].item()

            all_layers = list(range(num_layers))
            with QuickAblationHook(model, "attention", all_layers, ans_pos, family):
                ar = run_forward(model, tokenizer, prompt, cache, device)
            with QuickAblationHook(model, "mlp", all_layers, ans_pos, family):
                mr = run_forward(model, tokenizer, prompt, cache, device)

            attn_delta = abs(orig_logit - ar["logits"][0, -1, gold_id].item())
            mlp_delta = abs(orig_logit - mr["logits"][0, -1, gold_id].item())
            dominant = "MLP" if mlp_delta > attn_delta else "Attention"
            print(f"  {s['id']}: |Δ|_attn={attn_delta:.1f}, |Δ|_mlp={mlp_delta:.1f} → {dominant}")

    # ─── 3. S_X probe ───
    print("\n─── S_X: Residual State Probe ───")
    features, labels_list = [], []
    for s in samples:
        prompt = build_prompt(s)
        label = reasoning_type_to_label(s.get("reasoning_type", "direct_evidence"))
        if label < 0:
            continue
        r = run_forward(model, tokenizer, prompt, cache, device)
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

    if len(features) >= 10 and len(set(labels_list)) >= 2:
        fa = np.array(features)
        la = np.array(labels_list)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        mc = min(np.bincount(la))
        ns = min(3, mc)
        if ns >= 2 and len(fa) >= ns * len(set(la)):
            cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=42)
            acc = cross_val_score(clf, fa, la, cv=cv, scoring="accuracy").mean()
        else:
            clf.fit(fa, la)
            acc = clf.score(fa, la)
        s_x = compute_s_x(acc, len(set(la)))
        perm = run_permutation_test(fa, la, n_permutations=30)
        print(f"  Probe accuracy: {acc:.4f}, S_X: {s_x:.4f}")
        print(f"  Shuffled: {perm['shuffled_mean']:.4f}, p={perm['p_value']:.4f}")

    remove_hooks(cache)
    print("\nDone.")

    # Guidance for real dataset setup
    if not args.fever and not args.hotpotqa:
        print("\n─── Using controlled samples only. ───")
        print("To add real tasks, download:")
        print("  FEVER: https://fever.ai/resources.html")
        print("  HotpotQA: https://hotpotqa.github.io/")
        print("Then run:")
        print("  python experiments/run_real_task_validation.py --fever path/to/fever.jsonl")
        print("  python experiments/run_real_task_validation.py --hotpotqa path/to/hotpotqa.json")


if __name__ == "__main__":
    main()
