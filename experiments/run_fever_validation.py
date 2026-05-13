"""FEVER-style fact verification validation.

Tests the mechanism chain on fact verification formatted tasks:
- Claim + Evidence → SUPPORTS / REFUTES / NOT ENOUGH INFO
- Measures whether QK routes to evidence, whether residual encodes verdict,
  and whether MLP causally transforms evidence to verdict logit.

This validates that the mechanism is not an artifact of QA prompt formatting.

Usage:
    python experiments/run_fever_validation.py
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


def build_fever_prompt(sample: dict) -> str:
    """Build a FEVER-style fact verification prompt."""
    return (
        f"Evidence: {sample['evidence']}\n"
        f"Claim: {sample['claim']}\n"
        f"Verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO):"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading GPT-2...")
    model, tokenizer, cache, arch_info = load_model_v04("gpt2", device)
    num_layers = arch_info.get("num_layers", 12)

    # Load FEVER-style data
    data_path = Path(__file__).parent.parent / "data" / "fever_style.jsonl"
    samples = load_jsonl(data_path)
    print(f"Loaded {len(samples)} FEVER-style samples")
    labels = [s["label"] for s in samples]
    print(f"Label distribution: SUPPORTS={labels.count('SUPPORTS')}, "
          f"REFUTES={labels.count('REFUTES')}, NEI={labels.count('NOT ENOUGH INFO')}")

    # ─── 1. R_QK by verdict ───
    print("\n═══════════════════════════════════════")
    print("R_QK: Evidence Routing by Verdict Type")
    print("═══════════════════════════════════════")

    rqk_by_label = defaultdict(list)
    for s in samples:
        prompt = build_fever_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)

        # Find evidence span positions
        span = find_token_span(tokenizer, prompt, s["gold_evidence_span"])
        ev_pos = span["token_indices"]
        ans_pos = [len(r["tokens"]) - 1]  # last token (verdict position)

        rqk = compute_r_qk(r["attentions"], ans_pos, ev_pos)
        rqk_by_label[s["label"]].append(rqk)

    for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        vals = rqk_by_label.get(lbl, [])
        if vals:
            print(f"  {lbl:20s}: R_QK={sum(vals)/len(vals):.4f} (n={len(vals)})")

    # ─── 2. MLP vs Attention Ablation ───
    print("\n═══════════════════════════════════════")
    print("MLP vs Attention Ablation")
    print("═══════════════════════════════════════")

    from experiments.run_cross_model_foundation import QuickAblationHook

    for s in samples[:5]:
        prompt = build_fever_prompt(s)
        r = run_forward(model, tokenizer, prompt, cache, device)
        ans_pos = len(r["tokens"]) - 1

        # Get verdict token logit
        verdict_tokens = tokenizer.encode(s["label"], add_special_tokens=False)
        if not verdict_tokens:
            continue
        vt_id = verdict_tokens[0]
        orig_logit = r["logits"][0, -1, vt_id].item()

        all_layers = list(range(num_layers))
        with QuickAblationHook(model, "attention", all_layers, ans_pos, "gpt2"):
            ar = run_forward(model, tokenizer, prompt, cache, device)
        with QuickAblationHook(model, "mlp", all_layers, ans_pos, "gpt2"):
            mr = run_forward(model, tokenizer, prompt, cache, device)

        attn_d = abs(orig_logit - ar["logits"][0, -1, vt_id].item())
        mlp_d = abs(orig_logit - mr["logits"][0, -1, vt_id].item())
        dominant = "MLP" if mlp_d > attn_d else "Attn"
        print(f"  {s['id']} ({s['label']:>18s}): |Δ|_attn={attn_d:.1f}, |Δ|_mlp={mlp_d:.1f} → {dominant}")

    # ─── 3. S_X: Residual encoding of verdict ───
    print("\n═══════════════════════════════════════")
    print("S_X: Residual Encoding of Verdict Type")
    print("═══════════════════════════════════════")

    label_to_int = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    features, label_ints = [], []

    for s in samples:
        prompt = build_fever_prompt(s)
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
        print(f"  Classes: {num_cls}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  S_X: {s_x:.4f}")
        print(f"  Shuffled: {perm['shuffled_mean']:.4f}, p={perm['p_value']:.4f}")
        print(f"  Random baseline: {1.0/num_cls:.4f}")

    # ─── Summary ───
    print(f"\n═══════════════════════════════════════")
    print("FEVER-STYLE VALIDATION SUMMARY")
    print(f"═══════════════════════════════════════")

    supports_rqk = sum(rqk_by_label.get("SUPPORTS", [0])) / max(len(rqk_by_label.get("SUPPORTS", [1])), 1)
    refutes_rqk = sum(rqk_by_label.get("REFUTES", [0])) / max(len(rqk_by_label.get("REFUTES", [1])), 1)
    nei_rqk = sum(rqk_by_label.get("NOT ENOUGH INFO", [0])) / max(len(rqk_by_label.get("NOT ENOUGH INFO", [1])), 1)

    print(f"  R_QK: SUPPORTS={supports_rqk:.4f} | REFUTES={refutes_rqk:.4f} | NEI={nei_rqk:.4f}")
    print(f"  R_QK ratio (SUPPORTS/NEI): {supports_rqk/max(nei_rqk,1e-8):.1f}x")
    print(f"  S_X: {s_x:.4f} (p={perm['p_value']:.4f})")
    print(f"  MLP dominant in ablation: check per-sample above")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
