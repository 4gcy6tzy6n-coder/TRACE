"""Run ablation experiments: token ablation and attention masking.

Compares gold answer logit before and after interventions.

Usage:
    python experiments/run_ablation.py
    python experiments/run_ablation.py --limit 5
"""

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.causal_intervention import evidence_token_ablation, attention_evidence_ablation
from src.utils import load_jsonl, build_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)[: args.limit]

    results = []
    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample)
        print(f"\n[{idx + 1}/{len(samples)}] {sample['id']}")

        result = run_forward(model, tokenizer, prompt, cache, device)
        pos_info = get_evidence_token_positions(
            tokenizer, prompt, sample["evidence"], sample["gold_evidence_span"]
        )
        answer_positions = find_answer_position(
            tokenizer, result["tokens"], sample["gold_answer"]
        )
        evidence_positions = pos_info["gold_evidence_positions"]

        if not answer_positions:
            answer_positions = [len(result["tokens"]) - 1]

        # Token ablation
        token_result = evidence_token_ablation(
            model, tokenizer, cache, prompt,
            sample["gold_evidence_span"], sample["gold_answer"], device
        )
        print(f"  Token ablation: logit {token_result['original_logit']:.2f} → {token_result['ablated_logit']:.2f} (drop={token_result['logit_drop']:.2f})")

        # Attention ablation
        attn_result = attention_evidence_ablation(
            model, tokenizer, cache, prompt,
            evidence_positions, answer_positions, sample["gold_answer"], device
        )
        print(f"  Attn ablation:  logit {attn_result['original_logit']:.2f} → {attn_result['masked_logit']:.2f} (drop={attn_result['logit_drop']:.2f})")

        results.append({
            "sample_id": sample["id"],
            "reasoning_type": sample["reasoning_type"],
            "gold_answer": sample["gold_answer"],
            "C_token_ablation": token_result["C_token_ablation"],
            "C_attention_ablation": attn_result["C_attention_ablation"],
            "token_logit_drop": token_result["logit_drop"],
            "attn_logit_drop": attn_result["logit_drop"],
        })

    output_path = Path(__file__).parent.parent / "reports" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    # Summary
    avg_token_c = sum(r["C_token_ablation"] for r in results) / len(results)
    avg_attn_c = sum(r["C_attention_ablation"] for r in results) / len(results)
    print(f"\nAvg C_token_ablation: {avg_token_c:.4f}")
    print(f"Avg C_attention_ablation: {avg_attn_c:.4f}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
