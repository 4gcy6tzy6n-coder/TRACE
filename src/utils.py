"""Shared utilities for ICI evaluation.

v0.2: Added CoT/no-CoT prompt builders and heatmap data export.
"""

import json
import torch
from pathlib import Path
from typing import Optional


def load_jsonl(path: str | Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def normalize_score(raw: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip and normalize a score to [0, 1]."""
    return max(0.0, min(1.0, (raw - min_val) / (max_val - min_val + 1e-8)))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_prompt(sample: dict) -> str:
    """Build a full prompt from evidence + question for a sample."""
    evidence_text = "\n".join(sample["evidence"])
    return f"{evidence_text}\n\nQuestion: {sample['question']}\nAnswer:"


def build_cot_prompt(sample: dict) -> str:
    """Build prompt WITH chain-of-thought instruction."""
    evidence_text = "\n".join(sample["evidence"])
    return (
        f"{evidence_text}\n\n"
        f"Question: {sample['question']}\n"
        f"Let's think step by step.\n"
        f"Answer:"
    )


def build_no_cot_prompt(sample: dict) -> str:
    """Build prompt WITHOUT chain-of-thought instruction (direct answer)."""
    evidence_text = "\n".join(sample["evidence"])
    return (
        f"{evidence_text}\n\n"
        f"Question: {sample['question']}\n"
        f"Answer:"
    )


def export_heatmap_data(
    per_head_r_qk: dict[int, dict[int, float]],
    per_head_m_av: dict[int, dict[int, float]],
    layerwise_s_x: dict[int, float],
    output_path: str | Path,
) -> None:
    """Export layer/head scores as JSON for heatmap visualization.

    Output schema:
    {
        "num_layers": 12,
        "num_heads": 12,
        "R_QK": [[float x 12] x 12],
        "M_AV": [[float x 12] x 12],
        "S_X_per_layer": [float x 12]
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_layers = max(max(per_head_r_qk.keys(), default=-1),
                     max(per_head_m_av.keys(), default=-1)) + 1
    num_heads = 12  # GPT-2 standard

    r_qk_matrix = [[per_head_r_qk.get(l, {}).get(h, 0.0) for h in range(num_heads)]
                   for l in range(num_layers)]
    m_av_matrix = [[per_head_m_av.get(l, {}).get(h, 0.0) for h in range(num_heads)]
                   for l in range(num_layers)]
    s_x_per_layer = [layerwise_s_x.get(l, 0.0) for l in range(num_layers)]

    data = {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "R_QK": r_qk_matrix,
        "M_AV": m_av_matrix,
        "S_X_per_layer": s_x_per_layer,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
