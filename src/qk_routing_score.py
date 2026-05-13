"""R_QK: Measure whether attention routes to correct evidence.

v0.2: Added per-head analysis and evidence-routing head identification.
"""

import torch


def compute_r_qk(
    attentions: list[torch.Tensor],
    answer_positions: list[int],
    evidence_positions: list[int],
    normalize: bool = True,
) -> float:
    """Compute the R_QK score as attention mass from answer positions to evidence.

    R_QK = (1 / (L*H*|P|)) * sum_{l,h,p in P, j in E} A_{l,h}[p, j]

    Args:
        attentions: List of attention weight tensors, one per layer.
                    Each has shape [batch, heads, seq, seq].
        answer_positions: Token indices for answer/query positions p.
        evidence_positions: Token indices for evidence positions E.
        normalize: If True, normalize by number of answer positions.

    Returns:
        R_QK score in [0, 1].
    """
    if not answer_positions or not evidence_positions:
        return 0.0

    total_layers = len(attentions)
    if total_layers == 0:
        return 0.0

    layer_scores = []

    for layer_attn in attentions:
        # layer_attn: [batch, heads, seq, seq]
        layer_attn = layer_attn[0]  # remove batch dim
        num_heads = layer_attn.shape[0]

        head_scores = []
        for h in range(num_heads):
            mass_sum = 0.0
            for p in answer_positions:
                if p < layer_attn.shape[1]:
                    mass = layer_attn[h, p, evidence_positions].sum().item()
                    mass_sum += mass
            if len(answer_positions) > 0:
                head_scores.append(mass_sum / len(answer_positions))

        if head_scores:
            layer_scores.append(sum(head_scores) / len(head_scores))

    if not layer_scores:
        return 0.0

    raw = sum(layer_scores) / len(layer_scores)

    if normalize:
        # Clamp to reasonable range; attention mass per position is in [0, 1]
        return max(0.0, min(1.0, raw))

    return raw


def compute_r_qk_per_layer(
    attentions: list[torch.Tensor],
    answer_positions: list[int],
    evidence_positions: list[int],
) -> list[float]:
    """Compute per-layer R_QK scores for analysis."""
    scores = []
    for layer_attn in attentions:
        layer_attn = layer_attn[0]
        num_heads = layer_attn.shape[0]
        head_scores = []
        for h in range(num_heads):
            mass_sum = 0.0
            for p in answer_positions:
                if p < layer_attn.shape[1]:
                    mass = layer_attn[h, p, evidence_positions].sum().item()
                    mass_sum += mass
            if answer_positions:
                head_scores.append(mass_sum / len(answer_positions))
        scores.append(sum(head_scores) / len(head_scores) if head_scores else 0.0)
    return scores


def compute_r_qk_per_head(
    attentions: list[torch.Tensor],
    answer_positions: list[int],
    evidence_positions: list[int],
) -> dict[int, dict[int, float]]:
    """Compute R_QK for every (layer, head) pair.

    Args:
        attentions: List of [batch, heads, seq, seq] per layer.
        answer_positions: Token indices for answer positions.
        evidence_positions: Token indices for evidence positions.

    Returns:
        {layer_idx: {head_idx: score}} dict.
    """
    result: dict[int, dict[int, float]] = {}
    if not answer_positions or not evidence_positions:
        return result

    for l, layer_attn in enumerate(attentions):
        layer_attn = layer_attn[0]  # remove batch
        num_heads = layer_attn.shape[0]
        result[l] = {}
        for h in range(num_heads):
            mass_sum = 0.0
            for p in answer_positions:
                if p < layer_attn.shape[1]:
                    mass_sum += layer_attn[h, p, evidence_positions].sum().item()
            result[l][h] = mass_sum / len(answer_positions) if answer_positions else 0.0
    return result


def identify_evidence_routing_heads(
    per_head_r_qk: dict[int, dict[int, float]],
    threshold_percentile: float = 80.0,
) -> list[tuple[int, int, float]]:
    """Identify heads with highest R_QK scores.

    Args:
        per_head_r_qk: {layer: {head: score}} from compute_r_qk_per_head.
        threshold_percentile: Percentile threshold for inclusion (0-100).

    Returns:
        List of (layer, head, score) sorted by score descending.
    """
    all_scores: list[tuple[int, int, float]] = []
    for l, heads in per_head_r_qk.items():
        for h, score in heads.items():
            all_scores.append((l, h, score))
    all_scores.sort(key=lambda x: x[2], reverse=True)

    if threshold_percentile >= 100:
        return all_scores

    cutoff = sorted([s[2] for s in all_scores])[
        int(len(all_scores) * threshold_percentile / 100)
    ]
    return [(l, h, s) for l, h, s in all_scores if s >= cutoff]
