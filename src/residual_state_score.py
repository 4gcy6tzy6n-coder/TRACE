"""S_X: Measure whether residual stream encodes intermediate reasoning states.

v0.3: Added MLP activation separability analysis and permutation test controls.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
import numpy as np


def extract_residual_features(
    hidden_states: list[torch.Tensor],
    positions: list[int],
    layer: int = -1,
) -> np.ndarray:
    """Extract hidden state vectors at given positions for a specific layer.

    Args:
        hidden_states: List of [batch, seq, hidden_dim] tensors, one per layer.
        positions: Token positions to extract features from.
        layer: Layer index (-1 = last layer).

    Returns:
        numpy array of shape [len(positions), hidden_dim].
    """
    if not hidden_states or not positions:
        return np.array([])

    hs = hidden_states[layer][0]  # [seq, hidden_dim]
    features = []
    for pos in positions:
        if pos < hs.shape[0]:
            features.append(hs[pos].cpu().numpy())
    return np.array(features) if features else np.array([])


def extract_features_from_last_token(
    hidden_states: list[torch.Tensor],
    layer: int = -1,
) -> np.ndarray:
    """Extract the last token's hidden state as a summary feature.

    Args:
        hidden_states: List of [batch, seq, hidden_dim] tensors.
        layer: Layer index (-1 = last).

    Returns:
        numpy array of shape [1, hidden_dim].
    """
    hs = hidden_states[layer][0]  # [seq, hidden_dim]
    return hs[-1].cpu().float().numpy().reshape(1, -1)


def extract_multilayer_features(
    hidden_states: list[torch.Tensor],
    position: int,
    layers: list[int] | None = None,
) -> np.ndarray:
    """Extract hidden state at a specific position across multiple layers.

    Args:
        hidden_states: List of tensors, one per layer.
        position: Token position.
        layers: List of layer indices (default: all layers).

    Returns:
        numpy array of shape [len(layers) * hidden_dim] (concatenated).
    """
    if layers is None:
        layers = list(range(len(hidden_states)))

    features = []
    for l in layers:
        hs = hidden_states[l][0]
        if position < hs.shape[0]:
            features.append(hs[position].cpu().numpy())

    return np.concatenate(features) if features else np.array([])


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a logistic regression probe on residual stream features.

    Args:
        features: [N, D] feature matrix.
        labels: [N] integer labels.
        random_state: Random seed.

    Returns:
        Trained LogisticRegression model.
    """
    if len(features) == 0 or features.shape[0] != len(labels):
        raise ValueError(f"Feature/label mismatch: {features.shape} vs {labels.shape}")

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(features, labels)
    return clf


def compute_s_x(
    probe_accuracy: float,
    num_classes: int = 5,
) -> float:
    """Convert probe accuracy to S_X score normalized above random.

    S_X = (accuracy - random_baseline) / (1 - random_baseline)

    Args:
        probe_accuracy: Probe classification accuracy.
        num_classes: Number of reasoning type classes.

    Returns:
        S_X score in [0, 1] (clamped).
    """
    random_baseline = 1.0 / num_classes
    if probe_accuracy <= random_baseline:
        return 0.0
    if probe_accuracy >= 1.0:
        return 1.0
    s_x = (probe_accuracy - random_baseline) / (1.0 - random_baseline)
    return max(0.0, min(1.0, s_x))


def reasoning_type_to_label(reasoning_type: str) -> int:
    """Map reasoning type string to integer label."""
    mapping = {
        "direct_evidence": 0,
        "conflict": 1,
        "evidence_gap": 2,
        "misleading_hint": 3,
        "multi_step": 4,
    }
    return mapping.get(reasoning_type, -1)


# ─── v0.3: MLP Activation Analysis ───

def extract_mlp_features(
    mlp_outputs: list[torch.Tensor],
    position: int = -1,
    layer: int = -1,
) -> np.ndarray:
    """Extract MLP activation at a given position/layer.

    Args:
        mlp_outputs: List of [batch, seq, hidden] per layer.
        position: Token position (-1 = last).
        layer: Layer index (-1 = last).

    Returns:
        numpy array [1, hidden_dim].
    """
    if not mlp_outputs:
        return np.array([])
    mlp = mlp_outputs[layer][0]  # [seq, hidden]
    if position < 0:
        position = mlp.shape[0] + position
    return mlp[position].cpu().numpy().reshape(1, -1)


def train_mlp_probe(
    all_mlp_outputs: list[list[torch.Tensor]],
    labels: np.ndarray,
    num_layers: int = 12,
    n_splits: int = 5,
) -> dict[int, dict]:
    """Train linear probes on MLP activations (per layer), compare to residual probes.

    Returns:
        {layer_idx: {"mlp_accuracy": float, "residual_accuracy": float, "S_X_mlp": float}}
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    num_classes = len(set(labels))
    results = {}

    for layer in range(num_layers):
        features = []
        for sample_mlp in all_mlp_outputs:
            mlp = sample_mlp[layer][0]  # [seq, hidden]
            features.append(mlp[-1].cpu().numpy())

        features = np.array(features)
        if features.shape[0] < 5 or num_classes < 2:
            results[layer] = {"mlp_accuracy": 0.0, "S_X_mlp": 0.0}
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42)
        min_count = min(np.bincount(labels))
        fold_count = min(n_splits, min_count)
        if fold_count >= 2:
            cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
            acc = cv_scores.mean()
        else:
            clf.fit(features, labels)
            acc = clf.score(features, labels)

        s_x_mlp = compute_s_x(acc, num_classes)
        results[layer] = {
            "mlp_accuracy": round(float(acc), 4),
            "S_X_mlp": round(s_x_mlp, 4),
        }

    return results


def compare_residual_vs_mlp(
    residual_layerwise: dict[int, dict],
    mlp_layerwise: dict[int, dict],
) -> dict:
    """Compare residual stream probe accuracy to MLP activation probe accuracy.

    Returns:
        {"residual_avg_S_X": float, "mlp_avg_S_X": float, "per_layer": [...dict]}
    """
    per_layer = []
    residual_sx = []
    mlp_sx = []

    for layer in range(12):
        r_acc = residual_layerwise.get(layer, {}).get("accuracy", 0.0)
        m_acc = mlp_layerwise.get(layer, {}).get("mlp_accuracy", 0.0)
        r_sx = residual_layerwise.get(layer, {}).get("S_X", 0.0)
        m_sx = mlp_layerwise.get(layer, {}).get("S_X_mlp", 0.0)
        residual_sx.append(r_sx)
        mlp_sx.append(m_sx)
        per_layer.append({
            "layer": layer,
            "residual_accuracy": r_acc,
            "mlp_accuracy": m_acc,
            "residual_S_X": r_sx,
            "mlp_S_X": m_sx,
        })

    return {
        "residual_avg_S_X": sum(residual_sx) / len(residual_sx),
        "mlp_avg_S_X": sum(mlp_sx) / len(mlp_sx),
        "per_layer": per_layer,
    }


# ─── v0.3: Permutation Test Controls ───

def run_permutation_test(
    features: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 100,
    random_state: int = 42,
) -> dict:
    """Permutation test: shuffle labels and measure probe accuracy drop.

    If real accuracy >> shuffled accuracy distribution, residual stream
    genuinely encodes reasoning state rather than template features.

    Args:
        features: [N, D] feature matrix.
        labels: [N] integer labels.
        n_permutations: Number of shuffles.
        random_state: Random seed.

    Returns:
        {"real_accuracy": float, "shuffled_mean": float, "shuffled_std": float, "p_value": float}
    """
    import random
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    rng = np.random.RandomState(random_state)
    num_classes = len(set(labels))

    # Real accuracy
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    min_count = min(np.bincount(labels))
    n_splits = min(5, min_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        real_scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
        real_acc = real_scores.mean()
    else:
        clf.fit(features, labels)
        real_acc = clf.score(features, labels)

    # Shuffled accuracies
    shuffled_accs = []
    labels_copy = labels.copy()
    for _ in range(n_permutations):
        rng.shuffle(labels_copy)
        clf_perm = LogisticRegression(max_iter=1000, random_state=random_state)
        if n_splits >= 2:
            cv_perm = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            perm_scores = cross_val_score(clf_perm, features, labels_copy, cv=cv_perm, scoring="accuracy")
            shuffled_accs.append(perm_scores.mean())
        else:
            clf_perm.fit(features, labels_copy)
            shuffled_accs.append(clf_perm.score(features, labels_copy))

    shuffled_mean = float(np.mean(shuffled_accs))
    shuffled_std = float(np.std(shuffled_accs))

    # p-value: proportion of shuffled acc >= real acc
    p_value = sum(1 for a in shuffled_accs if a >= real_acc) / n_permutations

    return {
        "real_accuracy": round(float(real_acc), 4),
        "shuffled_mean": round(shuffled_mean, 4),
        "shuffled_std": round(shuffled_std, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def train_layerwise_probes(
    all_hidden_states: list[list[torch.Tensor]],
    labels: np.ndarray,
    num_layers: int = 12,
    n_splits: int = 5,
) -> dict[int, dict]:
    """Train a separate linear probe for each layer's hidden states.

    Uses the last token position from each layer. Returns per-layer accuracy
    and S_X under cross-validation.

    Args:
        all_hidden_states: List of per-sample hidden states lists.
            all_hidden_states[i][l] is [batch=1, seq, hidden] for sample i, layer l.
        labels: numpy array of integer labels [N].
        num_layers: Number of transformer layers.
        n_splits: CV fold count.

    Returns:
        {layer_idx: {"accuracy": float, "S_X": float, "std": float}}
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    num_classes = len(set(labels))
    results: dict[int, dict] = {}

    for layer in range(num_layers):
        features = []
        for sample_hs in all_hidden_states:
            hs = sample_hs[layer][0]  # [seq, hidden]
            features.append(hs[-1].cpu().numpy())  # last token

        features = np.array(features)
        if features.shape[0] < 5 or num_classes < 2:
            results[layer] = {"accuracy": 0.0, "S_X": 0.0, "std": 0.0}
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42)
        min_count = min(np.bincount(labels))
        fold_count = min(n_splits, min_count)
        if fold_count < 2:
            clf.fit(features, labels)
            acc = clf.score(features, labels)
            std = 0.0
        else:
            cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
            acc = cv_scores.mean()
            std = cv_scores.std()

        s_x = compute_s_x(acc, num_classes)
        results[layer] = {
            "accuracy": round(float(acc), 4),
            "S_X": round(s_x, 4),
            "std": round(float(std), 4),
        }

    return results
