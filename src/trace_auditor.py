"""TRACE: Transformer Reasoning Auditing through Causal Evidence traces.

Converts the mechanism chain (QK→MLP→X_l→logits) into an internal audit:
  1. IEAT construction — extract evidence-to-answer trace from internals
  2. Risk diagnosis — detect unsupported/misleading/hallucination risks
  3. Internal vs external comparison — is internal trace better than CoT?

The core claim: internal traces can detect black-box failures that external
signals (CoT, confidence) miss.
"""

import torch
import numpy as np
from typing import Optional


# ─── Risk types ───

RISK_TYPES = [
    "evidence_free",          # High logit, low R_QK — model not looking at evidence
    "misleading_driven",      # Misleading routing > evidence routing
    "unsupported_confidence", # S_X shows evidence_gap but logit is confident
    "distributed_uncertainty",# MLP/attention contributions dispersed, no stable state
    "low_internal_support",   # Overall ICI below threshold
]


def diagnose_risks(
    r_qk: float,
    m_av: float,
    s_x: float,
    c_do: float,
    ici: float,
    logit_confidence: float,
    reasoning_type_pred: Optional[str] = None,
    r_qk_misleading: Optional[float] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Diagnose black-box risks from internal trace signals.

    Args:
        r_qk: Evidence routing score.
        m_av: Evidence message contribution.
        s_x: Residual state separability.
        c_do: Causal intervention sensitivity.
        ici: Overall Internal CoT Index.
        logit_confidence: Softmax probability of top answer.
        reasoning_type_pred: Predicted reasoning type from residual probe.
        r_qk_misleading: R_QK to misleading tokens (if available).
        thresholds: Override default thresholds.

    Returns:
        dict with risk flags and overall risk level.
    """
    t = thresholds or {
        "r_qk_low": 0.05,
        "m_av_low": 0.1,
        "ici_low": 0.15,
        "r_qk_misleading_ratio": 1.5,
        "logit_confidence_high": 0.7,
    }

    risks = {}

    # Evidence-free: model answers confidently without looking at evidence
    risks["evidence_free"] = (
        r_qk < t["r_qk_low"] and logit_confidence > t["logit_confidence_high"]
    )

    # Misleading-driven: model routes more to misleading than to evidence
    if r_qk_misleading is not None and r_qk > 0:
        risks["misleading_driven"] = (
            r_qk_misleading / max(r_qk, 1e-8) > t["r_qk_misleading_ratio"]
        )
    else:
        risks["misleading_driven"] = False

    # Unsupported confidence: evidence_gap state but high confidence
    risks["unsupported_confidence"] = (
        reasoning_type_pred == "evidence_gap"
        and logit_confidence > t["logit_confidence_high"]
    )

    # Distributed uncertainty: MLP contribution low, state unclear
    risks["distributed_uncertainty"] = (
        m_av < t["m_av_low"] and r_qk < t["r_qk_low"]
    )

    # Low internal support: ICI below threshold
    risks["low_internal_support"] = ici < t["ici_low"]

    # Overall risk level: at least one risk flag
    active_risks = [k for k, v in risks.items() if v]
    risk_level = "high" if len(active_risks) >= 2 else (
        "medium" if len(active_risks) == 1 else "low"
    )

    return {
        "risks": risks,
        "active_risks": active_risks,
        "risk_level": risk_level,
        "risk_count": len(active_risks),
    }


def recommend_intervention(diagnosis: dict) -> dict:
    """Recommend intervention strategy based on risk diagnosis.

    Returns:
        {"action": str, "reason": str, "strength": str}
    """
    risks = diagnosis.get("risks", {})
    active = diagnosis.get("active_risks", [])

    if not active:
        return {
            "action": "none",
            "reason": "No internal risks detected. Answer appears evidence-grounded.",
            "strength": "none",
        }

    actions = []
    for risk in active:
        if risk == "evidence_free":
            actions.append({
                "action": "reformat",
                "reason": "Low evidence routing. Restructure prompt to separate evidence from query.",
                "strength": "strong",
            })
        elif risk == "misleading_driven":
            actions.append({
                "action": "filter_evidence",
                "reason": "Misleading evidence dominating routing. Remove or downweight misleading spans.",
                "strength": "strong",
            })
        elif risk == "unsupported_confidence":
            actions.append({
                "action": "abstain",
                "reason": "Evidence gap detected but model is confident. Recommend abstention.",
                "strength": "strong",
            })
        elif risk == "distributed_uncertainty":
            actions.append({
                "action": "verify",
                "reason": "Distributed low-contribution state. Verify answer with external retrieval.",
                "strength": "medium",
            })
        elif risk == "low_internal_support":
            actions.append({
                "action": "flag_unreliable",
                "reason": "Low overall internal support. Flag answer as potentially unreliable.",
                "strength": "medium",
            })

    # Return strongest action
    actions.sort(key=lambda a: {"strong": 2, "medium": 1, "none": 0}[a["strength"]], reverse=True)
    return actions[0] if actions else {"action": "none", "reason": "", "strength": "none"}


def compare_internal_vs_external(
    internal_risk: bool,
    external_signals: dict,
    ground_truth_error: bool,
) -> dict:
    """Compare internal trace vs external signals for error detection.

    Args:
        internal_risk: True if internal trace flags a risk.
        external_signals: {"cot_length": int, "self_consistency": float,
                           "logit_confidence": float, "attention_entropy": float}
        ground_truth_error: True if the model's answer is actually wrong.

    Returns:
        dict with detection metrics per signal.
    """
    signals = {}

    # Internal trace: risk flag
    signals["internal_trace"] = {
        "flagged": internal_risk,
        "correct_detection": internal_risk and ground_truth_error,
        "false_alarm": internal_risk and not ground_truth_error,
    }

    # Logit confidence: low confidence → predict error
    confidence = external_signals.get("logit_confidence", 0.5)
    low_conf = confidence < 0.5
    signals["logit_confidence"] = {
        "flagged": low_conf,
        "correct_detection": low_conf and ground_truth_error,
        "false_alarm": low_conf and not ground_truth_error,
    }

    # Attention entropy: high entropy → uncertain routing
    attn_entropy = external_signals.get("attention_entropy", 0.0)
    high_entropy = attn_entropy > 2.0
    signals["attention_entropy"] = {
        "flagged": high_entropy,
        "correct_detection": high_entropy and ground_truth_error,
        "false_alarm": high_entropy and not ground_truth_error,
    }

    return signals


def evaluate_detection_quality(comparisons: list[dict]) -> dict:
    """Compute precision, recall, F1 for each detection signal.

    Args:
        comparisons: List of results from compare_internal_vs_external.

    Returns:
        {"internal_trace": {"precision":, "recall":, "f1":}, ...}
    """
    metrics = {}
    signal_names = ["internal_trace", "logit_confidence", "attention_entropy"]

    for name in signal_names:
        tp = sum(1 for c in comparisons if c[name]["correct_detection"])
        fp = sum(1 for c in comparisons if c[name]["false_alarm"])
        fn = sum(1 for c in comparisons
                 if not c[name]["flagged"] and c[name].get("_ground_truth", False))

        # Need ground truth for fn — add it
        fn = 0
        for c in comparisons:
            is_error = c.get("_ground_truth", False)
            flagged = c[name]["flagged"]
            if is_error and not flagged:
                fn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        metrics[name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return metrics
