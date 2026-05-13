"""TRACE v3: Residual-State-Gated Trigger.

Upgrades trace-only trigger from:
  Trigger = f(R_QK, confidence)
to:
  Trigger = f(R_QK, confidence, S_X_state)

Uses a pre-trained residual probe to predict reasoning state,
then gates intervention based on predicted state + routing strength.
Goal: selective intervention without gold labels.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


class SXStatePredictor:
    """Predicts reasoning state from residual stream activations.

    Trained on controlled samples with known reasoning_type labels.
    At inference time, uses ONLY residual state — no gold labels.
    """

    def __init__(self):
        self.clf: LogisticRegression | None = None
        self.classes: list[str] = []
        self.trained = False

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Train logistic regression probe on residual features.

        Args:
            features: [N, hidden_dim] residual stream activations.
            labels: [N] integer labels (0=direct, 1=conflict, 2=gap, 3=misleading, 4=multi_step).
        """
        self.clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        self.clf.fit(features, labels)
        self.classes = ['direct_evidence', 'conflict', 'evidence_gap',
                        'misleading_hint', 'multi_step']
        self.trained = True

    def predict(self, feature: np.ndarray) -> tuple[str, float]:
        """Predict reasoning state from a single residual feature vector.

        Args:
            feature: [hidden_dim] residual activation.

        Returns:
            (predicted_state, confidence) tuple.
        """
        if not self.trained or self.clf is None:
            return ('unknown', 0.0)

        probs = self.clf.predict_proba(feature.reshape(1, -1))[0]
        pred_idx = probs.argmax()
        pred_state = self.classes[pred_idx] if pred_idx < len(self.classes) else 'unknown'
        confidence = float(probs[pred_idx])
        return (pred_state, confidence)


def trace_v3_decision(
    r_qk: float,
    logit_confidence: float,
    sx_state: str,
    sx_prob: float,
) -> tuple[str, str]:
    """TRACE v3 intervention decision based on internal signals only.

    Uses R_QK + logit confidence + S_X predicted state.
    NO gold reasoning_type is used.

    Args:
        r_qk: Evidence routing score.
        logit_confidence: Softmax probability of top token.
        sx_state: Predicted reasoning state from residual probe.
        sx_prob: Probe confidence for the predicted state.

    Returns:
        (action, reason) where action is one of:
        'none', 'conservative', 'disclose', 'filter'
    """
    # ─── Direct evidence: no intervention unless routing is catastrophically weak ───
    if sx_state == 'direct_evidence':
        if r_qk < 0.005 and logit_confidence < 0.2:
            return ('conservative', 'direct state but very weak routing + low confidence')
        return ('none', 'direct evidence state, routing adequate')

    # ─── Evidence gap: abstain ───
    if sx_state == 'evidence_gap':
        return ('conservative', 'evidence gap state detected')

    # ─── Conflict: disclose ───
    if sx_state == 'conflict':
        return ('conservative', 'conflict state detected, disclose')

    # ─── Misleading: conservative (filter is known ineffective) ───
    if sx_state == 'misleading_hint':
        if r_qk < 0.03:
            return ('conservative', 'misleading state + weak routing')
        return ('conservative', 'misleading state detected')

    # ─── Multi-step: conservative if routing very weak ───
    if sx_state == 'multi_step':
        if r_qk < 0.01 and logit_confidence < 0.3:
            return ('conservative', 'multi-step with weak evidence routing')
        return ('none', 'multi-step state, routing acceptable')

    # ─── Unknown/uncertain: conservative if routing clearly weak ───
    if r_qk < 0.015 and logit_confidence < 0.3:
        return ('conservative', 'uncertain state + weak routing')
    if r_qk < 0.01:
        return ('conservative', 'very weak routing, uncertain state')

    return ('none', 'no clear risk signal')
