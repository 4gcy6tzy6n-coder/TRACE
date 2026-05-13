"""Scale-aware ICI weighting and within-model normalization.

v0.5: Calibrates alpha/beta/gamma/delta per model scale rather than using
fixed 0.25 weights. Computes within-model R_QK gap/ratio for robust comparison.

Core insight: Small models rely on direct attention routing (R_QK, M_AV),
while larger models distribute reasoning across more layers and components
(S_X, distributed contribution).
"""

import torch
import numpy as np
from collections import defaultdict


def compute_within_model_rqk_gap(
    per_type_rqk: dict[str, list[float]],
) -> dict:
    """Compute within-model R_QK gap between direct_evidence and misleading_hint.

    Normalizes R_QK differences within each model, making cross-scale comparison
    robust to absolute attention mass differences.

    Args:
        per_type_rqk: {reasoning_type: [r_qk_values]} per sample.

    Returns:
        dict with gap, ratio, direct_mean, misleading_mean.
    """
    direct = per_type_rqk.get("direct_evidence", [])
    misleading = per_type_rqk.get("misleading_hint", [])

    if not direct or not misleading:
        return {"gap": 0.0, "ratio": 1.0, "direct_mean": 0.0, "misleading_mean": 0.0}

    direct_mean = sum(direct) / len(direct)
    misleading_mean = sum(misleading) / len(misleading)

    gap = direct_mean - misleading_mean
    ratio = direct_mean / max(misleading_mean, 1e-8)

    return {
        "gap": round(gap, 4),
        "ratio": round(ratio, 2),
        "direct_mean": round(direct_mean, 4),
        "misleading_mean": round(misleading_mean, 4),
    }


def compute_scale_aware_weights(
    rqk_gap: dict,
    s_x: float,
    m_av_range: dict[str, float],
    model_depth: int,
    model_heads: int,
) -> dict[str, float]:
    """Calibrate ICI weights based on model scale characteristics.

    Heuristics (empirically calibrated):
    - alpha (R_QK): higher for models with large R_QK gap → strong evidence routing
    - beta (M_AV): higher for models with strong direct evidence→answer contribution
    - gamma (S_X): higher for deeper models → more residual state encoding
    - delta (C_do): constant baseline, increases with patching sensitivity

    Args:
        rqk_gap: from compute_within_model_rqk_gap.
        s_x: global S_X probe score.
        m_av_range: {"direct_mean": float, "misleading_mean": float}.
        model_depth: number of layers.
        model_heads: number of attention heads.

    Returns:
        {"r_qk": alpha, "m_av": beta, "s_x": gamma, "c_do": delta}
    """
    # R_QK weight: proportional to gap (stronger routing → higher weight)
    alpha = min(0.40, max(0.15, 0.15 + rqk_gap.get("gap", 0) * 0.8))

    # M_AV weight: proportional to direct M_AV strength
    m_av_direct = m_av_range.get("direct_mean", 0.0)
    beta = min(0.40, max(0.10, m_av_direct * 0.35))

    # S_X weight: proportional to model depth (deeper → more state encoding)
    # Normalize: 6 layers → 0.15, 12 → 0.25, 24 → 0.35, 48 → 0.40
    gamma = min(0.40, max(0.10, 0.05 + model_depth * 0.012))

    # C_do weight: constant baseline, slightly higher for deeper models
    delta = min(0.30, max(0.10, 0.10 + model_depth * 0.003))

    # Normalize to sum to 1.0
    total = alpha + beta + gamma + delta
    weights = {
        "r_qk": round(alpha / total, 4),
        "m_av": round(beta / total, 4),
        "s_x": round(gamma / total, 4),
        "c_do": round(delta / total, 4),
    }

    return weights


def calibrate_weights_from_runs(
    model_results: list[dict],
) -> dict[str, dict]:
    """Calibrate scale-aware ICI weights for multiple models.

    Args:
        model_results: list of per-model evaluation results with keys:
            model_name, per_type (rqk, mav), S_X, arch_info.

    Returns:
        {model_name: {"weights": {...}, "rqk_gap": {...}}}
    """
    calibrated = {}

    for result in model_results:
        name = result["model_name"]
        per_type = result.get("per_type", {})

        # Extract per-type R_QK values
        per_type_rqk = {}
        per_type_mav = {}
        for rt, vals in per_type.items():
            if "rqk" in vals:
                per_type_rqk[rt] = [vals["rqk"]]  # aggregate values
            if "mav" in vals:
                per_type_mav[rt] = [vals["mav"]]

        rqk_gap = compute_within_model_rqk_gap(per_type_rqk)

        m_av_range = {
            "direct_mean": per_type.get("direct_evidence", {}).get("mav", 0.0),
            "misleading_mean": per_type.get("misleading_hint", {}).get("mav", 0.0),
        }

        arch = result.get("arch_info", {})
        depth = arch.get("num_layers", 12)
        heads = arch.get("num_heads", 12)
        s_x = result.get("S_X", 0.0)

        weights = compute_scale_aware_weights(
            rqk_gap, s_x, m_av_range, depth, heads
        )

        calibrated[name] = {
            "weights": weights,
            "rqk_gap": rqk_gap,
            "s_x": s_x,
            "model_depth": depth,
            "model_heads": heads,
        }

    return calibrated


def compute_scale_aware_ici(
    r_qk: float,
    m_av: float,
    s_x: float,
    c_do: float,
    weights: dict[str, float],
) -> float:
    """Compute ICI with scale-aware weights.

    Args:
        r_qk, m_av, s_x, c_do: Individual component scores.
        weights: {"r_qk": alpha, "m_av": beta, "s_x": gamma, "c_do": delta}.

    Returns:
        Scale-aware ICI score in [0, 1].
    """
    ici = (
        weights["r_qk"] * r_qk
        + weights["m_av"] * m_av
        + weights["s_x"] * s_x
        + weights["c_do"] * c_do
    )
    return max(0.0, min(1.0, ici))


def compare_fixed_vs_calibrated(
    per_model_results: dict[str, dict],
    calibrated_weights: dict[str, dict],
) -> dict:
    """Compare ICI rankings under fixed (0.25 each) vs scale-aware weights.

    Returns:
        {"fixed": {model: {type: ici}}, "calibrated": {model: {type: ici}}}
    """
    fixed_weights = {"r_qk": 0.25, "m_av": 0.25, "s_x": 0.25, "c_do": 0.25}
    comparison = {"fixed": {}, "calibrated": {}}

    for model_name, data in per_model_results.items():
        per_type = data.get("per_type", {})
        s_x = data.get("S_X", 0.0)
        calib_w = calibrated_weights.get(model_name, {}).get("weights", fixed_weights)

        fixed_model = {}
        calib_model = {}
        for rt, vals in per_type.items():
            rqk = vals.get("rqk", 0.0)
            mav = vals.get("mav", 0.0)
            fixed_model[rt] = round(
                compute_scale_aware_ici(rqk, mav, s_x, 0.0, fixed_weights), 4
            )
            calib_model[rt] = round(
                compute_scale_aware_ici(rqk, mav, s_x, 0.0, calib_w), 4
            )

        comparison["fixed"][model_name] = fixed_model
        comparison["calibrated"][model_name] = calib_model

    return comparison
