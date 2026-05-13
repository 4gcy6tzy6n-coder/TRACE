"""v0.6: MLP Causal Pathway Validation.

Three experiments:
1. MLP activation ablation — zero MLP_l[p] and measure answer logit drop
2. MLP patching — clean MLP → corrupted forward, measure answer recovery
3. Attention vs MLP causal comparison — which pathway matters more?

Closes the mechanism chain: QK routes → MLP transforms → residual stores → logits answer.

Usage:
    python experiments/run_v06_mlp_causal.py
    python experiments/run_v06_mlp_causal.py --limit 5
"""

import sys, json, argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_v04 import load_model_v04, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.utils import load_jsonl, build_prompt, ensure_dir
from src.activation_patching import ResidualPatchHook, _build_from_evidence


class MLPAblationHook:
    """Context manager that zeros MLP output at specific layers/positions."""

    def __init__(self, model, ablate_layers, ablate_position):
        self.model = model
        self.ablate_layers = set(ablate_layers or [])
        self.ablate_position = ablate_position
        self.handles = []

    def _make_hook(self, layer_idx):
        ablate_pos = self.ablate_position

        def hook(module, input, output):
            if ablate_pos is not None and ablate_pos < output.shape[1]:
                output[0, ablate_pos, :] = 0.0
            else:
                output[:] = 0.0
            return output

        return hook

    def __enter__(self):
        layers = self.ablate_layers or list(range(len(self.model.transformer.h)))
        for l in layers:
            block = self.model.transformer.h[l]
            handle = block.mlp.register_forward_hook(self._make_hook(l))
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def run_mlp_ablation(model, tokenizer, cache, prompt, gold_answer,
                     ablate_layers, ablate_position, device):
    """Zero MLP output at specified layers/position and measure logit drop."""
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    # Original
    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = orig_result["logits"][0, -1, gold_token_id].item()

    # Ablated
    with MLPAblationHook(model, ablate_layers, ablate_position):
        abl_result = run_forward(model, tokenizer, prompt, cache, device)
    abl_logit = abl_result["logits"][0, -1, gold_token_id].item()

    return {
        "original_logit": orig_logit,
        "ablated_logit": abl_logit,
        "logit_drop": orig_logit - abl_logit,
    }


def run_mlp_patching(model, tokenizer, cache, clean_prompt, corrupted_prompt,
                     gold_answer, patch_layers, device):
    """Patch clean MLP output into corrupted forward, measure recovery."""
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    # Clean run — collect MLP outputs
    clean_result = run_forward(model, tokenizer, clean_prompt, cache, device)
    clean_logit = clean_result["logits"][0, -1, gold_token_id].item()
    clean_mlp = [m.clone() for m in cache.mlp_outputs]

    # Corrupted run
    corr_result = run_forward(model, tokenizer, corrupted_prompt, cache, device)
    corr_logit = corr_result["logits"][0, -1, gold_token_id].item()

    # MLP patch: use ResidualPatchHook with mlp mode
    with ResidualPatchHook(
        model, clean_mlp, patch_layers=patch_layers,
        patch_mode="mlp", clean_mlp_outputs=clean_mlp,
    ):
        patched_result = run_forward(model, tokenizer, corrupted_prompt, cache, device)
    patched_logit = patched_result["logits"][0, -1, gold_token_id].item()

    clean_corr_gap = clean_logit - corr_logit
    if abs(clean_corr_gap) < 1e-8:
        recovery = 1.0
    else:
        recovery = max(0.0, min(2.0, (patched_logit - corr_logit) / clean_corr_gap))

    return {
        "clean_logit": clean_logit,
        "corrupted_logit": corr_logit,
        "patched_logit": patched_logit,
        "mlp_recovery_ratio": recovery,
    }


def run_attention_ablation_v06(model, tokenizer, cache, prompt, gold_answer,
                                ablate_layers, ablate_position, device):
    """Zero attention OUTPUT at specified layers/position (for comparison with MLP)."""
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = orig_result["logits"][0, -1, gold_token_id].item()

    class AttnAblationHook:
        def __init__(self, model_ref, ablate_layers, ablate_position):
            self.model_ref = model_ref
            self.ablate_layers = set(ablate_layers or [])
            self.ablate_position = ablate_position
            self.handles = []

        def _make_hook(self, layer_idx):
            pos = self.ablate_position

            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 1:
                    modified = list(output)
                    if pos is not None and pos < modified[0].shape[1]:
                        modified[0][0, pos, :] = 0.0
                    else:
                        modified[0][:] = 0.0
                    return tuple(modified)
                return output

            return hook

        def __enter__(self):
            for l in self.ablate_layers:
                handle = self.model_ref.transformer.h[l].attn.register_forward_hook(
                    self._make_hook(l))
                self.handles.append(handle)
            return self

        def __exit__(self, *args):
            for h in self.handles:
                h.remove()
            self.handles.clear()

    with AttnAblationHook(model, ablate_layers, ablate_position):
        abl_result = run_forward(model, tokenizer, prompt, cache, device)
    abl_logit = abl_result["logits"][0, -1, gold_token_id].item()

    return {
        "original_logit": orig_logit,
        "ablated_logit": abl_logit,
        "logit_drop": orig_logit - abl_logit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"v0.6 MLP Causal Pathway Validation — {args.model}")

    model, tokenizer, cache, arch_info = load_model_v04(args.model, device)
    num_layers = arch_info.get("num_layers", 12)
    print(f"Model: {arch_info['arch']}, {num_layers} layers")

    # Load direct evidence samples (clean, well-defined evidence routing)
    samples = [s for s in load_jsonl("data/toy_reasoning.jsonl")
               if s["reasoning_type"] == "direct_evidence"][:args.limit]

    # Also load corrupted pairs for patching
    pairs = load_jsonl("data/corrupted_pairs.jsonl")
    direct_pairs = [p for p in pairs if p["reasoning_type"] == "direct_evidence"][:args.limit]

    print(f"Evaluating {len(samples)} direct_evidence samples")
    print(f"Patching {len(direct_pairs)} corrupted pairs")

    # ─── Experiment 1: MLP Ablation ───
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: MLP Ablation (last token, all layers)")
    print("=" * 60)

    mlp_ablation_results = []
    for s in samples[:min(5, len(samples))]:
        prompt = build_prompt(s)
        result = run_forward(model, tokenizer, prompt, cache, device)
        ans_pos = len(result["tokens"]) - 1

        # Ablate MLP at answer position in ALL layers
        ab = run_mlp_ablation(
            model, tokenizer, cache, prompt, s["gold_answer"],
            ablate_layers=list(range(num_layers)),
            ablate_position=ans_pos, device=device,
        )
        mlp_ablation_results.append(ab)
        print(f"  {s['id']}: orig={ab['original_logit']:.2f} → mlp_abl={ab['ablated_logit']:.2f} (drop={ab['logit_drop']:+.2f})")

    avg_mlp_drop = sum(r["logit_drop"] for r in mlp_ablation_results) / len(mlp_ablation_results)
    print(f"  Avg MLP logit drop: {avg_mlp_drop:+.2f}")

    # ─── Experiment 2: Attention Ablation (for comparison) ───
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Attention Ablation (last token, all layers)")
    print("=" * 60)

    attn_ablation_results = []
    for s in samples[:min(5, len(samples))]:
        prompt = build_prompt(s)
        result = run_forward(model, tokenizer, prompt, cache, device)
        ans_pos = len(result["tokens"]) - 1

        ab = run_attention_ablation_v06(
            model, tokenizer, cache, prompt, s["gold_answer"],
            ablate_layers=list(range(num_layers)),
            ablate_position=ans_pos, device=device,
        )
        attn_ablation_results.append(ab)
        print(f"  {s['id']}: orig={ab['original_logit']:.2f} → attn_abl={ab['ablated_logit']:.2f} (drop={ab['logit_drop']:+.2f})")

    avg_attn_drop = sum(r["logit_drop"] for r in attn_ablation_results) / len(attn_ablation_results)
    print(f"  Avg Attn logit drop: {avg_attn_drop:+.2f}")

    # ─── Experiment 3: Layer-wise Ablation Sweep ───
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Layer-wise MLP vs Attention Ablation")
    print("=" * 60)

    s = samples[0]
    prompt = build_prompt(s)
    result = run_forward(model, tokenizer, prompt, cache, device)
    ans_pos = len(result["tokens"]) - 1

    print(f"\n  {'Layer':>6s} {'MLP Drop':>10s} {'Attn Drop':>10s} {'Dominant':>12s}")
    print(f"  {'-'*42}")

    layer_sweep = []
    for l in range(num_layers):
        mlp_ab = run_mlp_ablation(
            model, tokenizer, cache, prompt, s["gold_answer"],
            ablate_layers=[l], ablate_position=ans_pos, device=device,
        )
        attn_ab = run_attention_ablation_v06(
            model, tokenizer, cache, prompt, s["gold_answer"],
            ablate_layers=[l], ablate_position=ans_pos, device=device,
        )
        dominant = "MLP" if abs(mlp_ab["logit_drop"]) > abs(attn_ab["logit_drop"]) else "Attn"
        layer_sweep.append({
            "layer": l, "mlp_drop": mlp_ab["logit_drop"],
            "attn_drop": attn_ab["logit_drop"], "dominant": dominant,
        })
        print(f"  {l:6d} {mlp_ab['logit_drop']:+10.2f} {attn_ab['logit_drop']:+10.2f} {dominant:>12s}")

    # ─── Experiment 4: MLP Patching ───
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: MLP Patching (clean → corrupted)")
    print("=" * 60)

    mlp_patching_results = []
    for pair in direct_pairs[:min(5, len(direct_pairs))]:
        clean_prompt = _build_from_evidence(pair["clean_evidence"], pair["question"])
        corrupted_prompt = _build_from_evidence(pair["corrupted_evidence"], pair["question"])

        patch = run_mlp_patching(
            model, tokenizer, cache, clean_prompt, corrupted_prompt,
            pair["clean_answer"], patch_layers=list(range(num_layers)), device=device,
        )
        mlp_patching_results.append(patch)
        print(f"  {pair['pair_id']}: clean={patch['clean_logit']:.2f} corr={patch['corrupted_logit']:.2f} mlp_patch={patch['patched_logit']:.2f} recovery={patch['mlp_recovery_ratio']:.3f}")

    avg_mlp_recovery = sum(r["mlp_recovery_ratio"] for r in mlp_patching_results) / max(len(mlp_patching_results), 1)
    print(f"  Avg MLP patch recovery: {avg_mlp_recovery:.3f}")

    # ─── Causal Comparison Table ───
    print("\n" + "=" * 60)
    print("CAUSAL COMPARISON: Attention vs MLP")
    print("=" * 60)

    mlp_dominant_layers = sum(1 for l in layer_sweep if l["dominant"] == "MLP")
    attn_dominant_layers = sum(1 for l in layer_sweep if l["dominant"] == "Attn")

    comparison = {
        "model": args.model,
        "num_layers": num_layers,
        "avg_mlp_ablation_drop": round(float(avg_mlp_drop), 2),
        "avg_attn_ablation_drop": round(float(avg_attn_drop), 2),
        "mlp_dominant_layers": mlp_dominant_layers,
        "attn_dominant_layers": attn_dominant_layers,
        "avg_mlp_patch_recovery": round(float(avg_mlp_recovery), 3),
        "layer_sweep": layer_sweep,
    }

    print(f"  Avg MLP ablation drop:     {comparison['avg_mlp_ablation_drop']:+.2f}")
    print(f"  Avg Attn ablation drop:    {comparison['avg_attn_ablation_drop']:+.2f}")
    print(f"  MLP-dominant layers:       {mlp_dominant_layers}/{num_layers}")
    print(f"  Attn-dominant layers:      {attn_dominant_layers}/{num_layers}")
    print(f"  Avg MLP patch recovery:    {comparison['avg_mlp_patch_recovery']:.3f}")

    stronger = "MLP" if abs(avg_mlp_drop) > abs(avg_attn_drop) else "Attention"
    print(f"\n  Stronger causal pathway:   {stronger}")

    # Save
    output_path = Path(__file__).parent.parent / "reports" / "v06_mlp_causal.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
