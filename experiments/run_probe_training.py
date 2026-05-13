"""Train linear probes on residual stream hidden states.

Computes S_X by training a probe to classify reasoning_type from hidden states.

Usage:
    python experiments/run_probe_training.py
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.residual_state_score import (
    extract_features_from_last_token,
    train_linear_probe,
    compute_s_x,
    reasoning_type_to_label,
)
from src.utils import load_jsonl, build_prompt


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

    # Extract features and labels
    features = []
    labels = []

    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample)
        label = reasoning_type_to_label(sample["reasoning_type"])
        if label < 0:
            continue

        result = run_forward(model, tokenizer, prompt, cache, device)

        # Use last token hidden state from the final layer as feature
        feat = extract_features_from_last_token(result["hidden_states"], layer=-1)
        if feat.shape[0] > 0:
            features.append(feat.flatten())
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "direct_evidence", 1: "conflict", 2: "evidence_gap", 3: "misleading_hint", 4: "multi_step"}
    for u, c in zip(unique, counts):
        print(f"  {label_names[u]}: {c} samples")

    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    if num_classes < 2:
        print("\nCannot train probe: need at least 2 classes. S_X = 0.0")
        s_x = 0.0
        mean_acc = 0.0
        std_acc = 0.0
    else:
        # Train probe with cross-validation
        print("\nTraining linear probe with 5-fold CV...")
        clf = train_linear_probe(features, labels)

        n_splits = min(5, min(counts))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")

        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()
        print(f"CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # Compute S_X
        s_x = compute_s_x(mean_acc, num_classes)
        print(f"\nS_X = {s_x:.4f}")
        print(f"  (accuracy={mean_acc:.4f}, random_baseline={1.0/num_classes:.4f})")

    # Save probe results
    output_path = Path(__file__).parent.parent / "reports" / "probe_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "num_classes": num_classes,
        "num_samples": len(features),
        "cv_accuracy_mean": round(float(mean_acc), 4),
        "cv_accuracy_std": round(float(std_acc), 4),
        "random_baseline": round(1.0 / max(num_classes, 1), 4),
        "S_X": round(s_x, 4),
        "feature_dim": features.shape[1],
        "label_distribution": {label_names.get(u, str(u)): int(c) for u, c in zip(unique, counts)},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved probe results to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
