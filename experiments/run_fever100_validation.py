"""FEVER-100: Formal real-task mechanism validation with 3 format variants.

Tests: MLP dominance, R_QK format effect, S_X residual state encoding.
Formats: evidence-first, claim-first, QA-style.

Usage:
    python experiments/run_fever100_validation.py
"""

import sys, json
from pathlib import Path
from collections import defaultdict

import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import find_token_span
from src.qk_routing_score import compute_r_qk
from src.residual_state_score import (
    extract_features_from_last_token, compute_s_x, run_permutation_test,
)
from src.utils import load_jsonl, ensure_dir
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


# ─── Format builders ───

def build_evidence_first(sample):
    """Original FEVER: Evidence → Claim → Verdict"""
    return (f"Evidence: {sample['evidence']}\n"
            f"Claim: {sample['claim']}\n"
            f"Verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO):")

def build_claim_first(sample):
    """Claim-first: Claim → Evidence → Verdict"""
    return (f"Claim: {sample['claim']}\n"
            f"Evidence: {sample['evidence']}\n"
            f"Verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO):")

def build_qa_style(sample):
    """QA-style: Evidence + Question → Answer"""
    return (f"Evidence: {sample['evidence']}\n"
            f"Based on the evidence, is this claim true or false?\n"
            f"Claim: {sample['claim']}\n"
            f"Answer (SUPPORTS/REFUTES/NOT ENOUGH INFO):")


# ─── Ablation hook (minimal, same as cross-model) ───

class QuickAblationHook:
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

    def _get_mod(self, layer, names):
        for n in names:
            if hasattr(layer, n):
                return getattr(layer, n)
        return None

    def __enter__(self):
        for l in self.layers:
            if l >= len(self._layers):
                continue
            layer = self._layers[l]
            mod = (self._get_mod(layer, ('attn', 'self_attn')) if self.component == 'attention'
                   else self._get_mod(layer, ('mlp', 'feed_forward', 'ffn')))
            if mod:
                self.handles.append(mod.register_forward_hook(self._hook()))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self):
        pos = self.position
        comp = self.component
        def fn(module, input, output):
            if comp == 'attention' and isinstance(output, tuple) and len(output) >= 1:
                m = list(output)
                if pos is not None and pos < m[0].shape[1]:
                    m[0][0, pos, :] = 0.0
                else:
                    m[0][:] = 0.0
                return tuple(m)
            elif pos is not None and pos < output.shape[1]:
                out = output.clone()
                out[0, pos, :] = 0.0
                return out
            return torch.zeros_like(output)
        return fn


# ─── Main ───

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GPT-2...")
    model, tokenizer, cache, arch_info = load_model_v04("gpt2", device)
    num_layers = arch_info.get("num_layers", 12)
    family = arch_info.get("family", "gpt2")

    # Load FEVER-100
    samples = load_jsonl(Path(__file__).parent.parent / "data" / "fever_100.jsonl")
    labels = [s["label"] for s in samples]
    print(f"FEVER-100: {len(samples)} samples "
          f"(SUPPORTS={labels.count('SUPPORTS')}, "
          f"REFUTES={labels.count('REFUTES')}, "
          f"NEI={labels.count('NOT ENOUGH INFO')})")

    formats = {
        "evidence-first": build_evidence_first,
        "claim-first": build_claim_first,
        "qa-style": build_qa_style,
    }

    all_results = {}

    for fmt_name, fmt_fn in formats.items():
        print(f"\n{'='*60}")
        print(f"FORMAT: {fmt_name}")
        print(f"{'='*60}")

        # ─── R_QK by label ───
        print("\n  R_QK by verdict:")
        rqk_by_label = defaultdict(list)
        for s in samples:
            prompt = fmt_fn(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            span = find_token_span(tokenizer, prompt, s["gold_evidence_span"])
            ev_pos = span["token_indices"]
            ans_pos = [len(r["tokens"]) - 1]
            rqk = compute_r_qk(r["attentions"], ans_pos, ev_pos)
            rqk_by_label[s["label"]].append(rqk)

        rqk_summary = {}
        for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            vals = rqk_by_label.get(lbl, [])
            if vals:
                avg = sum(vals) / len(vals)
                rqk_summary[lbl] = round(avg, 4)
                print(f"    {lbl:20s}: R_QK={avg:.4f} (n={len(vals)})")

        # ─── MLP vs Attention Ablation ───
        print("\n  MLP vs Attention (first 10 SUPPORTS samples):")
        supports_samples = [s for s in samples if s["label"] == "SUPPORTS"][:10]
        mlp_wins = 0
        attn_wins = 0
        mlp_deltas = []
        attn_deltas = []

        for s in supports_samples:
            prompt = fmt_fn(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            ans_pos = len(r["tokens"]) - 1

            # Use "SUPPORTS" token as target
            target_id = tokenizer.encode("SUPPORTS", add_special_tokens=False)[0]
            orig_logit = r["logits"][0, -1, target_id].item()

            all_layers = list(range(num_layers))
            with QuickAblationHook(model, "attention", all_layers, ans_pos, family):
                ar = run_forward(model, tokenizer, prompt, cache, device)
            with QuickAblationHook(model, "mlp", all_layers, ans_pos, family):
                mr = run_forward(model, tokenizer, prompt, cache, device)

            ad = abs(orig_logit - ar["logits"][0, -1, target_id].item())
            md = abs(orig_logit - mr["logits"][0, -1, target_id].item())
            attn_deltas.append(ad)
            mlp_deltas.append(md)
            if md > ad:
                mlp_wins += 1
            else:
                attn_wins += 1

        avg_ad = sum(attn_deltas) / max(len(attn_deltas), 1)
        avg_md = sum(mlp_deltas) / max(len(mlp_deltas), 1)
        print(f"    MLP wins: {mlp_wins}/{mlp_wins+attn_wins}")
        print(f"    Avg |Δ|_attn={avg_ad:.1f}, Avg |Δ|_mlp={avg_md:.1f}, ratio={avg_md/max(avg_ad,1e-8):.2f}x")

        # ─── S_X residual probe ───
        print("\n  S_X: Residual encoding of verdict:")
        label_to_int = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        features, label_ints = [], []
        for s in samples:
            prompt = fmt_fn(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            feat = extract_features_from_last_token(r["hidden_states"], layer=-1)
            if feat.shape[0] > 0 and not np.isnan(feat).any():
                features.append(feat.flatten().astype(np.float32))
                label_ints.append(label_to_int[s["label"]])

        fa = np.array(features)
        la = np.array(label_ints)
        num_cls = len(set(la))

        if num_cls >= 2 and len(fa) >= 6:
            clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            mc = min(np.bincount(la))
            ns = min(3, mc)
            if ns >= 2 and len(fa) >= ns * num_cls:
                cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=42)
                acc = cross_val_score(clf, fa, la, cv=cv, scoring="accuracy").mean()
            else:
                clf.fit(fa, la)
                acc = clf.score(fa, la)
            s_x = compute_s_x(acc, num_cls)
            perm = run_permutation_test(fa, la, n_permutations=50)
            print(f"    Accuracy: {acc:.4f}, S_X: {s_x:.4f}")
            print(f"    Shuffled: {perm['shuffled_mean']:.4f}, p={perm['p_value']:.4f}")
            print(f"    Random baseline: {1.0/num_cls:.4f}")
        else:
            s_x = 0.0
            perm = {"shuffled_mean": 0, "p_value": 1.0}
            print(f"    Insufficient data for probe")

        all_results[fmt_name] = {
            "rqk": rqk_summary,
            "mlp_win_rate": round(mlp_wins / max(mlp_wins + attn_wins, 1), 3),
            "mlp_attn_ratio": round(avg_md / max(avg_ad, 1e-8), 2),
            "s_x": round(s_x, 4),
            "probe_accuracy": round(float(acc) if 'acc' in dir() else 0, 4),
            "p_value": perm["p_value"],
        }

    # ─── Cross-Format Summary ───
    print(f"\n{'='*70}")
    print("FEVER-100 CROSS-FORMAT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Format':20s} {'R_QK(S/R/NEI)':30s} {'MLP wins':>8s} {'MLP/Attn':>10s} {'S_X':>8s} {'p':>8s}")
    print(f"{'-'*85}")

    for fmt_name, r in all_results.items():
        rqk_str = f"{r['rqk'].get('SUPPORTS',0):.3f}/{r['rqk'].get('REFUTES',0):.3f}/{r['rqk'].get('NOT ENOUGH INFO',0):.3f}"
        print(f"{fmt_name:20s} {rqk_str:30s} {r['mlp_win_rate']:8.0%} {r['mlp_attn_ratio']:10.2f}x {r['s_x']:8.4f} {r['p_value']:8.4f}")

    # Key finding
    fmt_rqk_sep = {}
    for fmt_name, r in all_results.items():
        s_rqk = r['rqk'].get('SUPPORTS', 0)
        n_rqk = r['rqk'].get('NOT ENOUGH INFO', 1e-8)
        fmt_rqk_sep[fmt_name] = s_rqk / max(n_rqk, 1e-8) if n_rqk > 0 else 0

    best_fmt = max(fmt_rqk_sep, key=fmt_rqk_sep.get)
    print(f"\n  Best R_QK separation: {best_fmt} ({fmt_rqk_sep[best_fmt]:.1f}x SUPPORTS/NEI)")
    mlp_formats = sum(1 for r in all_results.values() if r['mlp_win_rate'] > 0.5)
    print(f"  MLP > Attention in {mlp_formats}/{len(formats)} formats")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "fever100_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
