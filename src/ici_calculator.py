"""ICI: Internal CoT Index — composite score combining all four metrics.

ICI = alpha * R_QK + beta * M_AV + gamma * S_X + delta * C_do
"""

import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    "r_qk": 0.25,
    "m_av": 0.25,
    "s_x": 0.25,
    "c_do": 0.25,
}


def compute_ici(
    r_qk: float,
    m_av: float,
    s_x: float,
    c_do: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the Internal CoT Index.

    Args:
        r_qk: Routing score (attention to evidence).
        m_av: Message score (evidence → answer).
        s_x: Residual state score (state encoding).
        c_do: Causal intervention score.
        weights: Dict with keys r_qk, m_av, s_x, c_do (default: equal).

    Returns:
        ICI score in [0, 1].
    """
    w = weights or DEFAULT_WEIGHTS
    ici = (
        w["r_qk"] * r_qk
        + w["m_av"] * m_av
        + w["s_x"] * s_x
        + w["c_do"] * c_do
    )
    return max(0.0, min(1.0, ici))


def compute_ici_for_sample(
    sample_id: str,
    r_qk: float,
    m_av: float,
    s_x: float,
    c_do: float,
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute ICI for a single sample and return full result dict.

    Args:
        sample_id: Sample identifier.
        r_qk, m_av, s_x, c_do: Individual scores.
        weights: Optional weight dict.

    Returns:
        dict with all scores.
    """
    ici = compute_ici(r_qk, m_av, s_x, c_do, weights)

    return {
        "sample_id": sample_id,
        "R_QK": round(r_qk, 4),
        "M_AV": round(m_av, 4),
        "S_X": round(s_x, 4),
        "C_do": round(c_do, 4),
        "ICI": round(ici, 4),
    }


def save_results(
    results: list[dict],
    path: str | Path,
) -> None:
    """Save ICI results to JSONL and JSON summary.

    Args:
        results: List of result dicts.
        path: Output path (JSONL format).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSONL
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary JSON
    summary_path = path.with_suffix(".summary.json")
    ici_scores = [r["ICI"] for r in results]
    summary = {
        "num_samples": len(results),
        "avg_ICI": round(sum(ici_scores) / len(ici_scores), 4) if ici_scores else 0,
        "min_ICI": round(min(ici_scores), 4) if ici_scores else 0,
        "max_ICI": round(max(ici_scores), 4) if ici_scores else 0,
        "avg_R_QK": round(sum(r["R_QK"] for r in results) / len(results), 4),
        "avg_M_AV": round(sum(r["M_AV"] for r in results) / len(results), 4),
        "avg_S_X": round(sum(r["S_X"] for r in results) / len(results), 4),
        "avg_C_do": round(sum(r["C_do"] for r in results) / len(results), 4),
        "weights": DEFAULT_WEIGHTS,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
