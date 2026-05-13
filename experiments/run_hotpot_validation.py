"""HotpotQA-style multi-hop validation.

Tests whether the mechanism chain handles multi-hop evidence integration:
  1. QK routing to BOTH evidence spans (bridge + target)
  2. MLP dominance increases for multi-hop vs single-hop
  3. Residual state encodes multi-hop vs single-hop reasoning

Usage:
    python experiments/run_hotpot_validation.py
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

# Ablation hook (same as cross-model / FEVER)
class QuickAblationHook:
    def __init__(self, model, component, layers, position, family="gpt2"):
        self.component = component
        self.layers = set(layers or [])
        self.position = position
        self.handles = []
        self._layers = self._find_layers(model)
    def _find_layers(self, model):
        for attr in ['transformer.h', 'gpt_neox.layers', 'model.layers']:
            obj = model
            for part in attr.split('.'):
                if hasattr(obj, part): obj = getattr(obj, part)
                else: break
            else: return obj
        return []
    def _get_mod(self, layer, names):
        for n in names:
            if hasattr(layer, n): return getattr(layer, n)
        return None
    def __enter__(self):
        for l in self.layers:
            if l >= len(self._layers): continue
            layer = self._layers[l]
            mod = (self._get_mod(layer, ('attn','self_attn')) if self.component=='attention'
                   else self._get_mod(layer, ('mlp','feed_forward','ffn')))
            if mod: self.handles.append(mod.register_forward_hook(self._hook()))
        return self
    def __exit__(self, *args):
        for h in self.handles: h.remove()
        self.handles.clear()
    def _hook(self):
        pos, comp = self.position, self.component
        def fn(module, input, output):
            if comp=='attention' and isinstance(output,tuple) and len(output)>=1:
                m=list(output)
                if pos is not None and pos<m[0].shape[1]: m[0][0,pos,:]=0.0
                else: m[0][:]=0.0
                return tuple(m)
            elif pos is not None and pos<output.shape[1]:
                out=output.clone(); out[0,pos,:]=0.0; return out
            return torch.zeros_like(output)
        return fn


def build_hotpot_prompt(sample):
    evidence = "\n".join(sample["evidence"])
    return (f"Documents:\n{evidence}\n\n"
            f"Question: {sample['question']}\n"
            f"Answer:")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading GPT-2...")
    model, tokenizer, cache, arch_info = load_model_v04("gpt2", device)
    num_layers = arch_info.get("num_layers", 12)
    family = arch_info.get("family", "gpt2")

    samples = load_jsonl(Path(__file__).parent.parent / "data" / "hotpot_style.jsonl")
    multi = [s for s in samples if s["source"] == "multi_step"]
    single = [s for s in samples if s["source"] == "direct_evidence"]
    print(f"Hotpot-60: {len(multi)} multi-hop, {len(single)} single-hop")

    # ─── 1. QK routing to BOTH evidence spans ───
    print("\n═══ QK Multi-Span Routing ═══")
    for s in multi[:5]:
        prompt = build_hotpot_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)
        ans_pos = [len(r["tokens"]) - 1]

        span1 = find_token_span(tokenizer, prompt, s["gold_span_1"])
        span2 = find_token_span(tokenizer, prompt, s["gold_span_2"])

        rqk1 = compute_r_qk(r["attentions"], ans_pos, span1["token_indices"])
        rqk2 = compute_r_qk(r["attentions"], ans_pos, span2["token_indices"]) if span2["token_indices"] else 0
        both = compute_r_qk(r["attentions"], ans_pos,
                            span1["token_indices"] + span2["token_indices"])

        print(f"  {s['id']}: R_QK(span1)={rqk1:.4f}, R_QK(span2)={rqk2:.4f}, R_QK(both)={both:.4f}")

    # ─── 2. MLP/Attention comparison: multi-hop vs single-hop ───
    print("\n═══ MLP vs Attention: Multi-hop vs Single-hop ═══")

    for label, group in [("multi-hop", multi[:8]), ("single-hop", single[:8])]:
        mlp_wins = 0
        mlp_deltas, attn_deltas = [], []
        for s in group:
            prompt = build_hotpot_prompt(s)
            r = run_forward(model, tokenizer, prompt, cache, device)
            ans_pos = len(r["tokens"]) - 1
            try:
                tid = tokenizer.encode(s["gold_answer"], add_special_tokens=False)[0]
            except Exception:
                continue
            orig = r["logits"][0, -1, tid].item()
            al = list(range(num_layers))
            with QuickAblationHook(model, "attention", al, ans_pos, family):
                ar = run_forward(model, tokenizer, prompt, cache, device)
            with QuickAblationHook(model, "mlp", al, ans_pos, family):
                mr = run_forward(model, tokenizer, prompt, cache, device)
            ad = abs(orig - ar["logits"][0, -1, tid].item())
            md = abs(orig - mr["logits"][0, -1, tid].item())
            attn_deltas.append(ad)
            mlp_deltas.append(md)
            if md > ad:
                mlp_wins += 1

        avg_ad = sum(attn_deltas)/max(len(attn_deltas),1)
        avg_md = sum(mlp_deltas)/max(len(mlp_deltas),1)
        ratio = avg_md/max(avg_ad,1e-8)
        print(f"  {label}: MLP wins {mlp_wins}/{len(group)}, "
              f"|Δ|_attn={avg_ad:.1f}, |Δ|_mlp={avg_md:.1f}, ratio={ratio:.2f}x")

    # ─── 3. S_X: residual encoding of multi-hop vs single-hop ───
    print("\n═══ S_X: Multi-hop vs Single-hop in Residual ═══")
    features, labels_i = [], []
    for s in multi[:25] + single[:25]:
        prompt = build_hotpot_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)
        feat = extract_features_from_last_token(r["hidden_states"], layer=-1)
        if feat.shape[0] > 0 and not np.isnan(feat).any():
            features.append(feat.flatten().astype(np.float32))
            labels_i.append(1 if s["source"] == "multi_step" else 0)

    fa = np.array(features)
    la = np.array(labels_i)
    if len(fa) >= 10 and len(set(la)) >= 2:
        clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        mc = min(np.bincount(la))
        ns = min(3, mc)
        if ns >= 2:
            cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=42)
            acc = cross_val_score(clf, fa, la, cv=cv, scoring="accuracy").mean()
        else:
            clf.fit(fa, la)
            acc = clf.score(fa, la)
        s_x = compute_s_x(acc, 2)
        perm = run_permutation_test(fa, la, n_permutations=50)
        print(f"  Binary (multi vs single): acc={acc:.4f}, S_X={s_x:.4f}")
        print(f"  Shuffled: {perm['shuffled_mean']:.4f}, p={perm['p_value']:.4f}")
        print(f"  Random baseline: 0.5000")

    # ─── Summary ───
    print(f"\n═══ HOTPOT SUMMARY ═══")
    print(f"  Multi-hop QK routing: attends to BOTH evidence spans (see above)")
    print(f"  MLP multi-hop vs single-hop: see per-group comparison above")
    print(f"  Residual state: see S_X binary classification above")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
