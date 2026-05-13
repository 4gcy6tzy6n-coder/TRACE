"""Run activation patching experiments on corrupted pairs.

Measures how much patching clean residual stream into corrupted forward
pass recovers the clean answer logit. Sweeps all layers individually.

Usage:
    python experiments/run_patching.py
    python experiments/run_patching.py --limit 5
    python experiments/run_patching.py --layer 6  # patch single layer
    python experiments/run_patching.py --all-layers  # patch all layers at once
"""

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, remove_hooks
from src.activation_patching import measure_logit_recovery, sweep_patching_layers
from src.utils import load_jsonl, build_prompt, ensure_dir


def build_prompt_from_evidence(evidence: list[str], question: str) -> str:
    """Build a prompt from evidence list and question."""
    evidence_text = "\n".join(evidence)
    return f"{evidence_text}\n\nQuestion: {question}\nAnswer:"


def main():
    parser = argparse.ArgumentParser(description="Run activation patching experiments")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs")
    parser.add_argument("--layer", type=int, default=-1, help="Patch specific layer only")
    parser.add_argument("--all-layers", action="store_true", help="Patch all layers at once")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    # Load corrupted pairs
    pairs_path = Path(__file__).parent.parent / "data" / "corrupted_pairs.jsonl"
    pairs = load_jsonl(pairs_path)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} corrupted pairs")

    results = []
    print("\n--- Running patching experiments ---")

    for idx, pair in enumerate(pairs):
        clean_prompt = build_prompt_from_evidence(
            pair["clean_evidence"], pair["question"]
        )
        corrupted_prompt = build_prompt_from_evidence(
            pair["corrupted_evidence"], pair["question"]
        )

        print(f"\n[{idx + 1}/{len(pairs)}] {pair['pair_id']} ({pair['reasoning_type']})")
        print(f"  Clean:     {pair['clean_answer']}")
        print(f"  Corrupted: {pair['corrupted_answer']}")

        if args.layer >= 0:
            # Single layer patching
            patch_layers = [args.layer]
            rec = measure_logit_recovery(
                model, tokenizer, cache, clean_prompt, corrupted_prompt,
                pair["clean_answer"], patch_layers=patch_layers, device=device,
            )
            print(f"  Layer {args.layer}: recovery={rec['recovery_ratio']:.4f}")
            results.append({
                "pair_id": pair["pair_id"],
                "reasoning_type": pair["reasoning_type"],
                "layer": args.layer,
                "recovery_ratio": rec["recovery_ratio"],
                "clean_logit": rec["clean_logit"],
                "corrupted_logit": rec["corrupted_logit"],
                "patched_logit": rec["patched_logit"],
            })
        elif args.all_layers:
            # Patch all layers
            rec = measure_logit_recovery(
                model, tokenizer, cache, clean_prompt, corrupted_prompt,
                pair["clean_answer"], patch_layers=None, device=device,
            )
            print(f"  All layers: recovery={rec['recovery_ratio']:.4f}")
            results.append({
                "pair_id": pair["pair_id"],
                "reasoning_type": pair["reasoning_type"],
                "layers": "all",
                "recovery_ratio": rec["recovery_ratio"],
                "clean_logit": rec["clean_logit"],
                "corrupted_logit": rec["corrupted_logit"],
                "patched_logit": rec["patched_logit"],
            })
        else:
            # Sweep all layers individually
            layer_results = sweep_patching_layers(
                model, tokenizer, cache, clean_prompt, corrupted_prompt,
                pair["clean_answer"], num_layers=12, device=device,
            )
            best_layer = max(layer_results, key=layer_results.get)
            print(f"  Best recovery: layer {best_layer} ({layer_results[best_layer]:.4f})")
            for l in range(12):
                if layer_results[l] > 0.01:
                    print(f"    L{l:02d}: {layer_results[l]:.4f}")

            results.append({
                "pair_id": pair["pair_id"],
                "reasoning_type": pair["reasoning_type"],
                "layer_sweep": layer_results,
                "best_layer": best_layer,
                "best_recovery": layer_results[best_layer],
            })

    # Save results
    output_dir = ensure_dir(Path(__file__).parent.parent / "reports")
    output_path = output_dir / "patching_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")

    # Summary
    if args.all_layers or args.layer >= 0:
        by_type = {}
        for r in results:
            rt = r["reasoning_type"]
            if rt not in by_type:
                by_type[rt] = []
            by_type[rt].append(r["recovery_ratio"])

        print("\n--- Summary ---")
        for rt, recs in by_type.items():
            avg = sum(recs) / len(recs)
            print(f"  {rt:20s}: avg recovery = {avg:.4f}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
