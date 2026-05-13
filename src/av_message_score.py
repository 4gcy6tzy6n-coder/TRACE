"""M_AV: Measure whether evidence value information drives the answer.

v0.2: True M_AV computed from A * V * W_O * W_U circuit.
v0.1 proxy (logit-drop) retained as compute_m_av_proxy() for baseline comparison.
"""

import torch
from src.model_loader import ActivationCache, run_forward, get_token_logit
from src.causal_intervention import _normalize_logit_drop


def compute_m_av_from_qkv(
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
    """True M_AV: evidence value flow through attention to answer logit.

    For each layer l and head h:
      message = sum_{j in E} A_{l,h}[p, j] * V_{l,h}[j]             [head_dim]
      projected = message @ W_O_head_slice                           [hidden_dim]
      logit_contrib = projected @ W_U[answer_token_id]                scalar

    M_AV is the normalized total logit contribution from evidence
    V vectors routed through attention to the answer position.

    Args:
        model: GPT-2 model.
        q/k/v_per_layer: Per-head Q/K/V per layer [batch, heads, seq, head_dim].
        evidence_positions: Token indices of gold evidence.
        answer_position: Token index of answer (or -1 for last).
        answer_token_id: Vocabulary ID of gold answer token.
        num_heads: Number of attention heads (12 for gpt2).
        head_dim: Dimension per head (64 for gpt2).

    Returns:
        dict with per_head_contributions, per_layer_aggregate,
        total_logit_contribution, M_AV.
    """
    hidden_dim = num_heads * head_dim  # 768
    seq_len = q_per_layer[0].shape[2]
    if answer_position < 0:
        answer_position = seq_len + answer_position
    # Clamp to valid range
    answer_position = max(0, min(answer_position, seq_len - 1))

    w_u = model.lm_head.weight.detach()  # [vocab, hidden_dim]
    num_layers = len(q_per_layer)
    evidence_set = [j for j in evidence_positions if 0 <= j < seq_len]

    per_layer_total = []
    per_head_all: dict[int, dict[int, float]] = {}

    for l in range(num_layers):
        q = q_per_layer[l][0]  # [heads, seq, head_dim]
        k = k_per_layer[l][0]
        v = v_per_layer[l][0]

        # Compute attention weights for this layer
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [heads, seq, seq]
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [heads, seq, seq]

        # W_O (c_proj) weight: [hidden_dim, hidden_dim] in Conv1D layout
        c_proj_w = model.transformer.h[l].attn.c_proj.weight.detach()  # [768, 768]

        per_head_all[l] = {}
        layer_total = 0.0

        for h in range(num_heads):
            attn_h = attn_weights[h]       # [seq, seq]
            v_h = v[h]                     # [seq, head_dim]

            # Message from evidence to answer: sum_j A[p, j] * V[j]
            message = torch.zeros(head_dim, device=q.device)
            total_weight = 0.0
            for j in evidence_set:
                a_pj = attn_h[answer_position, j]
                total_weight += a_pj
                message += a_pj * v_h[j]

            if total_weight < 1e-10:
                per_head_all[l][h] = 0.0
                continue

            # Project through W_O head slice
            # Head h occupies columns h*head_dim:(h+1)*head_dim in concat output
            # W_O weight is [768, 768]; head h's slice is rows h*head_dim:(h+1)*head_dim
            w_o_head = c_proj_w[h * head_dim : (h + 1) * head_dim, :]  # [head_dim, 768]
            projected = message @ w_o_head  # [768]

            # Project onto answer vocabulary token
            logit_contrib = torch.dot(projected, w_u[answer_token_id]).item()
            per_head_all[l][h] = logit_contrib
            layer_total += logit_contrib

        per_layer_total.append(layer_total)

    total_contribution = sum(per_layer_total)
    m_av = _normalize_logit_contribution(total_contribution)

    return {
        "per_head_contributions": per_head_all,
        "per_layer_aggregate": {l: v for l, v in enumerate(per_layer_total)},
        "total_logit_contribution": total_contribution,
        "M_AV": m_av,
    }


def compute_m_av_proxy(
    model,
    tokenizer,
    cache: ActivationCache,
    prompt: str,
    evidence_span: str,
    gold_answer: str,
    device: str = "cpu",
) -> dict:
    """v0.1 proxy: evidence removal logit drop (kept for baseline).

    This is the simplified method that removes the evidence span from
    the prompt text and measures logit change. Use compute_m_av_from_qkv
    for the true mechanistic M_AV.
    """
    gold_token_ids = tokenizer.encode(gold_answer, add_special_tokens=False)
    if not gold_token_ids:
        return {"original_logit": 0.0, "masked_logit": 0.0, "logit_drop": 0.0, "M_AV": 0.0}

    gold_token_id = gold_token_ids[0]

    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = get_token_logit(orig_result["logits"], gold_token_id)

    masked_prompt = prompt.replace(evidence_span, "")
    masked_prompt = " ".join(masked_prompt.split())

    mask_result = run_forward(model, tokenizer, masked_prompt, cache, device)
    masked_logit = get_token_logit(mask_result["logits"], gold_token_id)

    logit_drop = orig_logit - masked_logit
    m_av = _normalize_logit_drop(logit_drop)

    return {
        "original_logit": orig_logit,
        "masked_logit": masked_logit,
        "logit_drop": logit_drop,
        "M_AV": m_av,
    }


def compute_m_av_per_head_from_result(
    m_av_result: dict,
) -> dict[int, dict[int, float]]:
    """Extract per-head M_AV from the full computation output."""
    return m_av_result.get("per_head_contributions", {})


def _normalize_logit_contribution(
    raw_contribution: float,
    scale: float = 5.0,
) -> float:
    """Normalize raw logit contribution to [0, 1] M_AV score.

    Uses same sigmoid scaling as _normalize_logit_drop. Large positive
    contributions → near 1.0; near-zero or negative → near 0.0.
    """
    if raw_contribution <= 0:
        return 0.0
    return float(
        2.0 / (1.0 + torch.exp(torch.tensor(-raw_contribution / scale))) - 1.0
    )
