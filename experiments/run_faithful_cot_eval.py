"""Evaluate ICI on faithful vs unfaithful chain-of-thought samples.

Hypothesis: ICI(faithful CoT) > ICI(unfaithful CoT)
because faithful CoT traces align with internal evidence routing.

Usage:
    python experiments/run_faithful_cot_eval.py
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.token_mapper import get_evidence_token_positions, find_answer_position
from src.qk_routing_score import compute_r_qk
from src.av_message_score import compute_m_av_from_qkv
from src.ici_calculator import compute_ici_for_sample
from src.utils import load_jsonl, ensure_dir


def build_cot_prompt_from_fields(evidence: list[str], question: str, cot_text: str) -> str:
    """Build prompt with a specific CoT text."""
    evidence_text = "\n".join(evidence)
    return f"{evidence_text}\n\nQuestion: {question}\nLet's think step by step.\n{cot_text}\nAnswer:"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    # Load faithful/unfaithful CoT pairs
    pairs_path = Path(__file__).parent.parent / "data" / "faithful_unfaithful_cot.jsonl"
    pairs = load_jsonl(pairs_path)
    print(f"Loaded {len(pairs)} faithful/unfaithful pairs")

    results = []
    faithful_icis = []
    unfaithful_icis = []

    for pair in pairs:
        print(f"\n{pair['id']} ({pair['faithfulness_label']} baseline)")

        # Build prompts with faithful and unfaithful CoT
        faithful_prompt = build_cot_prompt_from_fields(
            pair["evidence"], pair["question"], pair["faithful_cot"]
        )
        unfaithful_prompt = build_cot_prompt_from_fields(
            pair["evidence"], pair["question"], pair["unfaithful_cot"]
        )

        # Run forward on both
        f_result = run_forward(model, tokenizer, faithful_prompt, cache, device)
        u_result = run_forward(model, tokenizer, unfaithful_prompt, cache, device)

        # Get evidence positions (same evidence for both)
        pos_info = get_evidence_token_positions(
            tokenizer, faithful_prompt, pair["evidence"],
            pair["evidence"][0].split(": ")[1] if ": " in pair["evidence"][0] else pair["evidence"][0]
        )
        ev_pos = pos_info["gold_evidence_positions"]
        if not ev_pos:
            ev_pos = pos_info["all_evidence_positions"]

        # Answer positions and token IDs
        gold_token_id = tokenizer.encode(pair["gold_answer"], add_special_tokens=False)[0]

        # Answer position = last token (use attention matrix seq_len for robustness)
        f_seq_len = f_result["attentions"][0].shape[-1]
        f_ans_pos = [f_seq_len - 1]
        f_rqk = compute_r_qk(f_result["attentions"], f_ans_pos, ev_pos)
        f_mav = compute_m_av_from_qkv(
            model, f_result["q_per_layer"], f_result["k_per_layer"],
            f_result["v_per_layer"], ev_pos, f_seq_len - 1, gold_token_id,
        )["M_AV"]

        # Unfaithful: R_QK and M_AV
        u_seq_len = u_result["attentions"][0].shape[-1]
        u_ans_pos = [u_seq_len - 1]
        u_rqk = compute_r_qk(u_result["attentions"], u_ans_pos, ev_pos)
        u_mav = compute_m_av_from_qkv(
            model, u_result["q_per_layer"], u_result["k_per_layer"],
            u_result["v_per_layer"], ev_pos, u_seq_len - 1, gold_token_id,
        )["M_AV"]

        # ICI (C_do and S_X omitted for speed; focus on R_QK + M_AV)
        f_ici = compute_ici_for_sample(pair["id"], f_rqk, f_mav, 0.0, 0.0,
                                       weights={"r_qk": 0.5, "m_av": 0.5, "s_x": 0.0, "c_do": 0.0})
        u_ici = compute_ici_for_sample(pair["id"], u_rqk, u_mav, 0.0, 0.0,
                                       weights={"r_qk": 0.5, "m_av": 0.5, "s_x": 0.0, "c_do": 0.0})

        print(f"  Faithful:   R_QK={f_rqk:.4f}, M_AV={f_mav:.4f}, ICI={f_ici['ICI']:.4f}")
        print(f"  Unfaithful: R_QK={u_rqk:.4f}, M_AV={u_mav:.4f}, ICI={u_ici['ICI']:.4f}")

        results.append({
            "id": pair["id"],
            "faithful_R_QK": round(f_rqk, 4),
            "faithful_M_AV": round(f_mav, 4),
            "faithful_ICI": f_ici["ICI"],
            "unfaithful_R_QK": round(u_rqk, 4),
            "unfaithful_M_AV": round(u_mav, 4),
            "unfaithful_ICI": u_ici["ICI"],
            "ICI_diff": round(f_ici["ICI"] - u_ici["ICI"], 4),
        })
        faithful_icis.append(f_ici["ICI"])
        unfaithful_icis.append(u_ici["ICI"])

    # Summary
    avg_faithful = sum(faithful_icis) / len(faithful_icis)
    avg_unfaithful = sum(unfaithful_icis) / len(unfaithful_icis)
    wins = sum(1 for r in results if r["ICI_diff"] > 0)
    losses = sum(1 for r in results if r["ICI_diff"] < 0)

    print(f"\n=== Faithful vs Unfaithful CoT Summary ===")
    print(f"  Avg ICI (faithful):   {avg_faithful:.4f}")
    print(f"  Avg ICI (unfaithful): {avg_unfaithful:.4f}")
    print(f"  Difference:           {avg_faithful - avg_unfaithful:+.4f}")
    print(f"  Faithful > Unfaithful: {wins}/{len(results)} pairs")

    # Save
    output_dir = ensure_dir(Path(__file__).parent.parent / "reports")
    output_path = output_dir / "faithful_cot_results.json"
    summary = {
        "num_pairs": len(results),
        "avg_ICI_faithful": round(avg_faithful, 4),
        "avg_ICI_unfaithful": round(avg_unfaithful, 4),
        "faithful_wins": wins,
        "unfaithful_wins": losses,
        "per_pair": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
