"""Statistical reporting for TRACE-Scale error reduction results.

Provides Wilson confidence intervals, paired bootstrap CIs, and McNemar tests
for Nature-level statistical rigor.
"""

import numpy as np
from scipy import stats as scipy_stats


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Correct for proportions near 0 or 1; recommended over normal approximation.

    Args:
        successes: Number of successes (e.g., errors).
        n: Total trials.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper, observed_rate) as floats.
    """
    if n == 0:
        return (0.0, 1.0, 0.0)

    p = successes / n
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper, p)


def format_rate_ci(successes: int, n: int, confidence: float = 0.95, pct: bool = True) -> str:
    """Format a rate with Wilson CI: '70.0% [61.2, 78.8]'.

    Args:
        successes: Number of successes.
        n: Total trials.
        confidence: Confidence level.
        pct: If True, format as percentage; if False, as proportion.

    Returns:
        Formatted string.
    """
    lo, hi, rate = wilson_ci(successes, n, confidence)
    if pct:
        return f"{rate*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
    else:
        return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def paired_bootstrap_ci(
    raw_outcomes: list[bool],
    trace_outcomes: list[bool],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    paired: bool = True,
) -> dict:
    """Bootstrap confidence interval for absolute and relative error reduction.

    Uses paired resampling (sample-level) by default.

    Args:
        raw_outcomes: List of bools (True = error) for raw model.
        trace_outcomes: List of bools (True = error) for TRACE model.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level.
        paired: If True, resample sample indices (paired); if False, resample independently.

    Returns:
        dict with absolute_reduction, relative_reduction, and their CIs.
    """
    n = len(raw_outcomes)
    if n == 0:
        return {"abs_reduction": 0, "abs_ci": (0, 0), "rel_reduction": 0, "rel_ci": (0, 0)}

    raw_rate = sum(raw_outcomes) / n
    trace_rate = sum(trace_outcomes) / n
    abs_red = raw_rate - trace_rate
    rel_red = abs_red / max(raw_rate, 1e-8)

    abs_diffs = []
    rel_diffs = []

    rng = np.random.RandomState(42)
    indices = np.arange(n)

    for _ in range(n_bootstrap):
        if paired:
            boot_idx = rng.choice(indices, size=n, replace=True)
            boot_raw = sum(raw_outcomes[i] for i in boot_idx) / n
            boot_trace = sum(trace_outcomes[i] for i in boot_idx) / n
        else:
            boot_raw = sum(rng.choice(raw_outcomes, size=n, replace=True)) / n
            boot_trace = sum(rng.choice(trace_outcomes, size=n, replace=True)) / n

        boot_abs = boot_raw - boot_trace
        boot_rel = boot_abs / max(boot_raw, 1e-8)
        abs_diffs.append(boot_abs)
        rel_diffs.append(boot_rel)

    alpha = 1 - confidence
    abs_lo = np.percentile(abs_diffs, alpha / 2 * 100)
    abs_hi = np.percentile(abs_diffs, (1 - alpha / 2) * 100)
    rel_lo = np.percentile(rel_diffs, alpha / 2 * 100)
    rel_hi = np.percentile(rel_diffs, (1 - alpha / 2) * 100)

    return {
        "raw_rate": raw_rate,
        "trace_rate": trace_rate,
        "abs_reduction": abs_red,
        "abs_ci": (abs_lo, abs_hi),
        "rel_reduction": rel_red,
        "rel_ci": (rel_lo, rel_hi),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def mcnemar_test(
    raw_errors: list[bool],
    trace_errors: list[bool],
) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether the proportion of errors differs between raw and TRACE.

    Args:
        raw_errors: Per-sample error flags for raw model.
        trace_errors: Per-sample error flags for TRACE model.

    Returns:
        dict with statistic, p_value, b, c counts.
    """
    b = 0  # raw error, trace correct
    c = 0  # raw correct, trace error

    for r, t in zip(raw_errors, trace_errors):
        if r and not t:
            b += 1
        elif not r and t:
            c += 1

    n_discordant = b + c
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "n_discordant": 0}

    # McNemar with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = 1 - scipy_stats.chi2.cdf(statistic, 1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "b": b,  # raw error → trace correct
        "c": c,  # raw correct → trace error
        "n_discordant": n_discordant,
        "significant_001": p_value < 0.001,
        "significant_005": p_value < 0.05,
    }


def format_reduction_ci(result: dict, as_pct: bool = True) -> str:
    """Format bootstrap reduction result as a readable string.

    Example: '75.0% [61.8%, 84.5%]'
    """
    abs_lo, abs_hi = result["abs_ci"]
    rel_lo, rel_hi = result["rel_ci"]

    mult = 100 if as_pct else 1
    unit = "%" if as_pct else ""

    abs_str = f"{result['abs_reduction']*mult:.1f}{unit} [{abs_lo*mult:.1f}{unit}, {abs_hi*mult:.1f}{unit}]"
    rel_str = f"{result['rel_reduction']*mult:.1f}{unit} [{rel_lo*mult:.1f}{unit}, {rel_hi*mult:.1f}{unit}]"

    return f"Absolute: {abs_str}\nRelative: {rel_str}"


def generate_statistical_table(
    per_type_results: dict,
    confidence: float = 0.95,
) -> str:
    """Generate a full statistical results table in markdown.

    Args:
        per_type_results: dict with keys like 'conflict', 'evidence_gap', etc.
            Each value: {"n": int, "raw_errors": int, "trace_errors": int,
                         "raw_outcomes": list[bool], "trace_outcomes": list[bool]}

    Returns:
        Markdown-formatted table string.
    """
    header = (
        "| Type | n | Raw Error [95% CI] | TRACE Error [95% CI] | "
        "Abs. Reduction [95% CI] | Rel. Reduction [95% CI] | McNemar p |"
    )
    sep = "|" + "|".join([" --- " for _ in range(7)]) + "|"

    rows = [header, sep]

    for type_name, data in per_type_results.items():
        n = data["n"]
        raw_errs = data["raw_errors"]
        trace_errs = data["trace_errors"]

        raw_rate_str = format_rate_ci(raw_errs, n, confidence)
        trace_rate_str = format_rate_ci(trace_errs, n, confidence)

        boot = paired_bootstrap_ci(
            data["raw_outcomes"], data["trace_outcomes"],
            n_bootstrap=10000, confidence=confidence,
        )
        abs_lo, abs_hi = boot["abs_ci"]
        rel_lo, rel_hi = boot["rel_ci"]

        abs_str = f"{boot['abs_reduction']*100:.1f}pp [{abs_lo*100:.1f}, {abs_hi*100:.1f}]"
        rel_str = f"{boot['rel_reduction']*100:.1f}% [{rel_lo*100:.1f}, {rel_hi*100:.1f}]"

        mcn = mcnemar_test(data["raw_outcomes"], data["trace_outcomes"])
        p_str = f"{mcn['p_value']:.4f}" if mcn['p_value'] >= 0.0001 else "<0.0001"
        if mcn["significant_001"]:
            p_str += "***"
        elif mcn["significant_005"]:
            p_str += "*"

        row = (
            f"| {type_name} | {n} | {raw_rate_str} | {trace_rate_str} | "
            f"{abs_str} | {rel_str} | {p_str} |"
        )
        rows.append(row)

    return "\n".join(rows)
