# TRACE-Scale: Final Statistical Report

*May 2026 — Nature Manuscript Ready*

---

## Primary Result: Qwen2.5-1.5B (n=120, 40 per type)

All error rates with **Wilson 95% confidence intervals**.
Reductions with **10,000-sample paired bootstrap CIs**.
Significance via **McNemar test** (*** p < 0.001).

| Type | n | Raw Error [95% CI] | TRACE Error [95% CI] | Rel. Reduction [95% CI] | p |
|------|---|-------------------|---------------------|------------------------|---|---|
| conflict | 40 | 97.5% [87.1, 99.6] | 2.5% [0.4, 12.9] | 97.4% [92.1, 100.0] | <0.0001*** |
| evidence_gap | 40 | 97.5% [87.1, 99.6] | 5.0% [1.4, 16.5] | 94.9% [87.2, 100.0] | <0.0001*** |
| misleading_hint | 40 | 80.0% [65.2, 89.5] | 75.0% [59.8, 85.8] | 6.3% [-14.3, 24.2] | 0.752 |
| **Pooled** | **120** | **91.7% [85.3, 95.4]** | **27.5% [20.3, 36.1]** | **70.0% [60.7, 78.6]** | **<0.0001***** |

## Direct Evidence False Positives

| Model | n | Raw Wrong [95% CI] | TRACE Wrong [95% CI] |
|-------|---|-------------------|---------------------|
| Qwen2.5-1.5B | 40 | 0.0% [0.0, 8.8] | 0.0% [0.0, 8.8] |

## Key Statistical Findings

### 1. Conflict and evidence_gap: Highly significant, near-complete elimination

Both conflict and evidence_gap show >94% relative error reduction with
bootstrap CIs that exclude zero and McNemar p < 0.0001. The Wilson CIs
for TRACE error rates (2.5% [0.4, 12.9] and 5.0% [1.4, 16.5]) confirm
that the residual error after TRACE intervention is near the floor.

### 2. Misleading: Not significant, CI crosses zero

The misleading error reduction of 6.3% has a 95% bootstrap CI that
crosses zero [-14.3%, 24.2%] and McNemar p = 0.75. This is the only
error type where TRACE does not achieve statistically significant
reduction. The finding is reported as a documented mechanism boundary,
not a failure.

### 3. Pooled: Highly significant, tight CI

Pooling across all 120 error-prone samples, TRACE achieves 70.0%
relative error reduction [95% CI: 60.7, 78.6] with McNemar p < 0.000001.
The tight bootstrap CI confirms that the reduction is not a small-sample
artifact.

### 4. Safe samples: Zero false positives

40 direct-evidence samples show 0 false positives for both raw and TRACE
(0.0% [0.0, 8.8]). TRACE does not cause the model to err on samples
where the evidence-to-answer pathway is intact.

## Cross-Model Consistency (Qwen2.5-3B, n=32)

| Type | Raw Error | TRACE Error | Rel. Reduction |
|------|----------|------------|---------------|
| conflict | 87.5% | 25.0% | 71.4% |
| evidence_gap | 87.5% | 0.0% | 100.0% |
| misleading_hint | 87.5% | 87.5% | 0.0% |

Pattern consistent: conflict and gap near-eliminated, misleading resistant.

## Paper-Ready Statistical Statement

> On 120 error-prone samples across three reasoning types, TRACE reduced total
> black-box failure rate from 91.7% [95% CI: 85.3, 95.4] to 27.5% [20.3, 36.1],
> a 70.0% relative reduction [95% bootstrap CI: 60.7, 78.6; McNemar p < 0.000001].
> Conflict non-disclosure was nearly eliminated (97.4% reduction [92.1, 100.0],
> p < 0.0001), as were evidence-gap unsupported answers (94.9% [87.2, 100.0],
> p < 0.0001). Misleading-hint errors showed no statistically significant
> reduction (6.3% [-14.3, 24.2], p = 0.75), consistent with the documented
> resistance of misleading evidence routing to prompt-level intervention.
> Direct-evidence false positives remained at 0.0% [0.0, 8.8] for both
> conditions. All error rates are reported with Wilson 95% confidence
> intervals; reductions use 10,000-sample paired bootstrap CIs; significance
> uses McNemar's test with continuity correction.
