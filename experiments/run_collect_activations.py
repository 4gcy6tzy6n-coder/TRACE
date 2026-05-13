"""Collect activations for all samples and save them for later analysis.

Usage:
    python experiments/run_collect_activations.py
"""

import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.utils import load_jsonl, build_prompt, ensure_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)
    print(f"Loaded {len(samples)} samples")

    # Collect activations
    output_dir = ensure_dir(Path(__file__).parent.parent / "data" / "activations")

    results = []
    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample)
        print(f"\n[{idx + 1}/{len(samples)}] {sample['id']}")

        # Run forward
        result = run_forward(model, tokenizer, prompt, cache, device)

        # Map evidence and answer positions
        pos_info = get_evidence_token_positions(
            tokenizer, prompt, sample["evidence"], sample["gold_evidence_span"]
        )
        answer_positions = find_answer_position(
            tokenizer, result["tokens"], sample["gold_answer"]
        )

        # Save activation summary
        activation_data = {
            "sample_id": sample["id"],
            "prompt": prompt,
            "tokens": result["tokens"],
            "num_layers": len(result["hidden_states"]),
            "hidden_dim": result["hidden_states"][0].shape[-1],
            "seq_len": len(result["tokens"]),
            "gold_evidence_positions": pos_info["gold_evidence_positions"],
            "answer_positions": answer_positions,
            "reasoning_type": sample["reasoning_type"],
            "label": sample["label"],
        }
        results.append(activation_data)

        # Save hidden states and attentions for this sample
        torch.save(
            {
                "hidden_states": [hs.cpu() for hs in result["hidden_states"]],
                "attentions": [attn.cpu() for attn in result["attentions"]],
                "logits": result["logits"].cpu(),
            },
            output_dir / f"{sample['id']}_activations.pt",
        )

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved activations to {output_dir}")

    # Cleanup
    remove_hooks(cache)


if __name__ == "__main__":
    main()
