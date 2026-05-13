"""Distributed M_AV: Multi-pathway evidence-to-answer contribution analysis.

v0.5: Decomposes evidence contribution into:
- Attention-pathway: evidence V through attention heads to answer
- MLP-pathway: evidence information through MLP layers to answer
- Layer-wise: contribution breakdown per layer
- Head-wise: contribution breakdown per head within each layer

Addresses the v0.4 finding that M_AV decreases in larger models by
distributing the measurement across all internal pathways.
"""

import torch
import numpy as np


def compute_attention_pathway_contribution(
    model,
    q_per_layer: list[torch.Tensor],
    k_per_layer: list[torch.Tensor],
    v_per_layer: list[torch.Tensor],
    evidence_positions: list[int],
    answer_position: int,
    answer_token_id: int,
    num_heads: int = 12,
    head_dim: int = 64,
) -> dict:
    """Attention-only pathway: evidence V through attention to answer logit.

    For each layer l and head h:
      A_{l,h} = softmax(QK^T/sqrt(d))
      message = sum_{j in E} A[p,j] * V[j]
      projected = message @ W_O_head_slice
      logit_contrib = projected @ W_U[answer_token]
    """
    hidden_dim = num_heads * head_dim
    seq_len = q_per_layer[0].shape[2]
    if answer_position < 0:
        answer_position = seq_len + answer_position
    answer_position = max(0, min(answer_position, seq_len - 1))

    w_u = model.lm_head.weight.detach()
    num_layers = len(q_per_layer)
    evidence_set = [j for j in evidence_positions if 0 <= j < seq_len]

    per_layer = {}
    per_head_all: dict[int, dict[int, float]] = {}
    total_contrib = 0.0

    for l in range(num_layers):
        q = q_per_layer[l][0]
        k = k_per_layer[l][0]
        v = v_per_layer[l][0]

        # Skip NaN layers
        if torch.isnan(q).any() or torch.isnan(v).any():
            per_layer[l] = 0.0
            per_head_all[l] = {}
            continue

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        c_proj_w = model.transformer.h[l].attn.c_proj.weight.detach()

        per_head_all[l] = {}
        layer_total = 0.0

        for h in range(num_heads):
            attn_h = attn_weights[h]
            v_h = v[h]

            message = torch.zeros(head_dim, device=q.device)
            total_weight = 0.0
            for j in evidence_set:
                a_pj = attn_h[answer_position, j]
                total_weight += a_pj
                message += a_pj * v_h[j]

            if total_weight < 1e-10:
                per_head_all[l][h] = 0.0
                continue

            w_o_head = c_proj_w[h * head_dim:(h + 1) * head_dim, :]
            projected = message @ w_o_head
            logit_contrib = torch.dot(projected, w_u[answer_token_id]).item()
            per_head_all[l][h] = logit_contrib
            layer_total += logit_contrib

        per_layer[l] = layer_total
        total_contrib += layer_total

    return {
        "per_layer": per_layer,
        "per_head": per_head_all,
        "total_attention_contribution": total_contrib,
        "top_contributing_layers": sorted(per_layer.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
    }


def compute_mlp_pathway_contribution(
    mlp_outputs: list[torch.Tensor],
    answer_token_id: int,
    answer_position: int = -1,
    w_u: torch.Tensor | None = None,
) -> dict:
    """MLP-pathway: MLP activation at answer position projected to vocabulary.

    This measures how much the MLP output at the answer position
    contributes to the answer token logit (before adding to residual).

    Note: This is a partial measurement — MLP contribution also flows
    through subsequent attention layers. For full pathway, see
    compute_distributed_contribution.
    """
    if w_u is None:
        w_u = torch.eye(mlp_outputs[0].shape[-1])  # identity fallback

    per_layer = {}
    for l, mlp_out in enumerate(mlp_outputs):
        seq_len = mlp_out.shape[1]
        pos = answer_position if answer_position >= 0 else seq_len + answer_position
        pos = max(0, min(pos, seq_len - 1))

        mlp_vec = mlp_out[0, pos]  # [hidden_dim]
        if torch.isnan(mlp_vec).any():
            per_layer[l] = 0.0
            continue

        contrib = torch.dot(mlp_vec.float(), w_u[answer_token_id].float()).item()
        per_layer[l] = contrib

    return {
        "per_layer_mlp": per_layer,
        "total_mlp_contribution": sum(per_layer.values()),
    }


def compute_distributed_contribution(
    model,
    q_per_layer: list[torch.Tensor],
    k_per_layer: list[torch.Tensor],
    v_per_layer: list[torch.Tensor],
    mlp_outputs: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    evidence_positions: list[int],
    answer_position: int,
    answer_token_id: int,
    num_heads: int = 12,
    head_dim: int = 64,
) -> dict:
    """Full distributed contribution: attention + MLP + residual pathways.

    Decomposes the total evidence-to-answer contribution into:
    1. Attention pathway (A×V×W_O)
    2. MLP pathway (direct MLP output projection)
    3. Residual pathway (total hidden state projection)
    4. Per-layer breakdown of all three
    """
    # Attention pathway
    attn = compute_attention_pathway_contribution(
        model, q_per_layer, k_per_layer, v_per_layer,
        evidence_positions, answer_position, answer_token_id,
        num_heads, head_dim,
    )

    # MLP pathway
    w_u = model.lm_head.weight.detach()
    mlp = compute_mlp_pathway_contribution(
        mlp_outputs, answer_token_id, answer_position, w_u,
    )

    # Residual pathway: total hidden state at answer position → vocabulary
    seq_len = hidden_states[-1].shape[1]
    pos = answer_position if answer_position >= 0 else seq_len + answer_position
    pos = max(0, min(pos, seq_len - 1))

    residual_per_layer = {}
    for l, hs in enumerate(hidden_states):
        hs_vec = hs[0, pos]
        if torch.isnan(hs_vec).any():
            residual_per_layer[l] = 0.0
            continue
        contrib = torch.dot(hs_vec.float(), w_u[answer_token_id].float()).item()
        residual_per_layer[l] = contrib

    # Compute fractions
    total_attn = attn["total_attention_contribution"]
    total_mlp = mlp["total_mlp_contribution"]
    total_residual = sum(residual_per_layer.values())
    total = abs(total_attn) + abs(total_mlp) + abs(total_residual) + 1e-8

    return {
        "attention_pathway": {
            "total": total_attn,
            "fraction": round(abs(total_attn) / total, 4),
            "per_layer": attn["per_layer"],
            "per_head": attn["per_head"],
            "top_layers": attn["top_contributing_layers"],
        },
        "mlp_pathway": {
            "total": total_mlp,
            "fraction": round(abs(total_mlp) / total, 4),
            "per_layer": mlp["per_layer_mlp"],
        },
        "residual_pathway": {
            "total": total_residual,
            "fraction": round(abs(total_residual) / total, 4),
            "per_layer": residual_per_layer,
        },
        "dominant_pathway": (
            "attention" if abs(total_attn) >= abs(total_mlp) and abs(total_attn) >= abs(total_residual)
            else "mlp" if abs(total_mlp) >= abs(total_residual)
            else "residual"
        ),
    }


def compute_pathway_switch_score(
    distributed_results: list[dict],
) -> dict:
    """Analyze whether the dominant pathway shifts with model scale.

    Computes the fraction of samples where attention dominates vs MLP dominates.
    """
    attn_dominant = sum(1 for r in distributed_results if r.get("dominant_pathway") == "attention")
    mlp_dominant = sum(1 for r in distributed_results if r.get("dominant_pathway") == "mlp")
    residual_dominant = sum(1 for r in distributed_results if r.get("dominant_pathway") == "residual")
    total = len(distributed_results)

    attn_fracs = [r["attention_pathway"]["fraction"] for r in distributed_results]
    mlp_fracs = [r["mlp_pathway"]["fraction"] for r in distributed_results]

    return {
        "attention_dominant_pct": round(attn_dominant / max(total, 1) * 100, 1),
        "mlp_dominant_pct": round(mlp_dominant / max(total, 1) * 100, 1),
        "residual_dominant_pct": round(residual_dominant / max(total, 1) * 100, 1),
        "avg_attention_fraction": round(sum(attn_fracs) / max(len(attn_fracs), 1), 4),
        "avg_mlp_fraction": round(sum(mlp_fracs) / max(len(mlp_fracs), 1), 4),
    }
