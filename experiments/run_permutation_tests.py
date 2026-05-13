"""Run permutation tests for S_X probe robustness.

Validates that residual stream state encoding is genuine:
1. Real labels vs shuffled labels
2. Real hidden states vs random states
3. Bag-of-token baseline vs hidden states

Usage:
    python experiments/run_permutation_tests.py
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model, run_forward, remove_hooks
from src.residual_state_score import (
    extract_features_from_last_token,
    train_linear_probe,
    compute_s_x,
    reasoning_type_to_label,
    run_permutation_test,
)
from src.utils import load_jsonl, build_prompt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model, tokenizer, cache = load_model("gpt2", device)

    # Load all 200 samples
    data_path = Path(__file__).parent.parent / "data" / "toy_reasoning.jsonl"
    samples = load_jsonl(data_path)
    print(f"Loaded {len(samples)} samples")

    # Extract features and labels
    features = []
    labels = []
    tokens_per_sample = []

    for sample in samples:
        prompt = build_prompt(sample)
        label = reasoning_type_to_label(sample["reasoning_type"])
        if label < 0:
            continue
        result = run_forward(model, tokenizer, prompt, cache, device)
        feat = extract_features_from_last_token(result["hidden_states"], layer=-1)
        if feat.shape[0] > 0:
            features.append(feat.flatten())
            labels.append(label)
            tokens_per_sample.append(len(result["tokens"]))

    features = np.array(features)
    labels = np.array(labels)
    num_classes = len(set(labels))
    print(f"\nFeatures: {features.shape}, Classes: {num_classes}")

    results = {}

    # ─── Test 1: Label Permutation ───
    print("\n=== Test 1: Label Permutation ===")
    perm_result = run_permutation_test(features, labels, n_permutations=100)
    print(f"  Real accuracy:     {perm_result['real_accuracy']:.4f}")
    print(f"  Shuffled mean:     {perm_result['shuffled_mean']:.4f} ± {perm_result['shuffled_std']:.4f}")
    print(f"  p-value:           {perm_result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {perm_result['significant']}")
    results["label_permutation"] = perm_result

    # ─── Test 2: Random Hidden States ───
    print("\n=== Test 2: Random Hidden States ===")
    random_feats = np.random.randn(*features.shape).astype(np.float32)
    clf_rand = train_linear_probe(random_feats, labels)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    min_count = min(np.bincount(labels))
    n_splits = min(5, min_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        rand_scores = cross_val_score(clf_rand, random_feats, labels, cv=cv, scoring="accuracy")
        rand_acc = rand_scores.mean()
    else:
        clf_rand.fit(random_feats, labels)
        rand_acc = clf_rand.score(random_feats, labels)
    random_s_x = compute_s_x(rand_acc, num_classes)
    print(f"  Random states accuracy: {rand_acc:.4f}")
    print(f"  Random states S_X:      {random_s_x:.4f}")
    results["random_states"] = {"accuracy": round(float(rand_acc), 4), "S_X": round(random_s_x, 4)}

    # ─── Test 3: Bag-of-Tokens Baseline ───
    print("\n=== Test 3: Token Count Baseline ===")
    token_counts = np.array(tokens_per_sample).reshape(-1, 1).astype(np.float32)
    # Add a few random dims to match feature dim
    token_feats = np.hstack([token_counts, np.random.randn(len(token_counts), features.shape[1] - 1).astype(np.float32)])
    clf_tok = train_linear_probe(token_feats, labels)
    if n_splits >= 2:
        cv2 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        tok_scores = cross_val_score(clf_tok, token_feats, labels, cv=cv2, scoring="accuracy")
        tok_acc = tok_scores.mean()
    else:
        clf_tok.fit(token_feats, labels)
        tok_acc = clf_tok.score(token_feats, labels)
    tok_s_x = compute_s_x(tok_acc, num_classes)
    print(f"  Token count accuracy: {tok_acc:.4f}")
    print(f"  Token count S_X:      {tok_s_x:.4f}")
    results["token_count_baseline"] = {"accuracy": round(float(tok_acc), 4), "S_X": round(tok_s_x, 4)}

    # ─── Test 4: Real Residual Probe ───
    print("\n=== Test 4: Real Residual Probe ===")
    clf = train_linear_probe(features, labels)
    if n_splits >= 2:
        cv3 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        real_scores = cross_val_score(clf, features, labels, cv=cv3, scoring="accuracy")
        real_acc = real_scores.mean()
        real_std = real_scores.std()
    else:
        clf.fit(features, labels)
        real_acc = clf.score(features, labels)
        real_std = 0.0
    real_s_x = compute_s_x(real_acc, num_classes)
    print(f"  Residual probe accuracy: {real_acc:.4f} ± {real_std:.4f}")
    print(f"  Residual S_X:            {real_s_x:.4f}")
    results["residual_probe"] = {"accuracy": round(float(real_acc), 4), "std": round(float(real_std), 4), "S_X": round(real_s_x, 4)}

    # ─── Summary Table ───
    print("\n=== S_X Robustness Summary ===")
    print(f"  {'Probe Setup':30s} {'Accuracy':>10s} {'S_X':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'Real residual states':30s} {results['residual_probe']['accuracy']:10.4f} {results['residual_probe']['S_X']:8.4f}")
    print(f"  {'Random hidden states':30s} {results['random_states']['accuracy']:10.4f} {results['random_states']['S_X']:8.4f}")
    print(f"  {'Token count baseline':30s} {results['token_count_baseline']['accuracy']:10.4f} {results['token_count_baseline']['S_X']:8.4f}")
    print(f"  {'Shuffled labels (mean)':30s} {results['label_permutation']['shuffled_mean']:10.4f} {'-':>8s}")

    # Save results
    output_path = Path(__file__).parent.parent / "reports" / "permutation_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    remove_hooks(cache)


if __name__ == "__main__":
    main()
