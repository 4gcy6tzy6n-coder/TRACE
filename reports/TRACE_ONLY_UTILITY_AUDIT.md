# TRACE-Only Utility Audit

*May 2026*

---

## Result Summary

| Metric | Value | Assessment |
|--------|-------|-----------|
| Gold-label independence | **✓ Proven** | TRACE triggers without reasoning_type |
| Fire rate | **92.5%** | Too high — near blanket |
| Error reduction | **76.2% → 6.2%** | Strong but at utility cost |
| Direct evidence retained | **17→4 (/20)** | Severe over-abstention |
| Over-abstention | **13/80** | 13 correct answers lost |
| Wrong→Fixed | **57/80** | 57 errors resolved |

## Per-Type Fire Rate

| Type | Fire Rate | Avg R_QK | Avg Confidence |
|------|----------|---------|---------------|
| conflict | 100% | 0.012 | 0.459 |
| evidence_gap | 100% | 0.012 | 0.270 |
| misleading_hint | 95% | 0.013 | 0.344 |
| direct_evidence | 75% | 0.026 | 0.432 |

## Output Classification

| Setting | Correct | Abstain | Conflict | Wrong |
|---------|---------|---------|----------|-------|
| Raw | 18/80 | 1/80 | 0/80 | 61/80 |
| TRACE | 4/80 | 71/80 | 0/80 | 5/80 |

## Direct Evidence Utility

- Raw correct: 17/20
- Trace correct: 4/20
- Trace abstentions: 13/20
- **65% of direct evidence correct answers became abstentions**

## Honest Assessment

### What This Proves

1. **Gold-label independence**: TRACE can trigger interventions using only internal
   signals (R_QK + confidence), without reasoning_type labels. Audit risk resolved.

2. **Error reduction is real**: 57/80 wrong answers were fixed. Only 5 errors remain.

### What This Reveals

1. **Current trigger is too conservative**: 92.5% fire rate means nearly all samples
   receive conservative intervention. This is blanket conservatism, not selective.

2. **Severe over-abstention**: 13/20 direct evidence correct answers became
   "Cannot determine." Utility cost is unacceptable for deployment.

3. **Simple R_QK thresholds insufficient**: Qwen2.5-1.5B has uniformly low R_QK
   (0.01–0.05), making R_QK-based discrimination between "needs intervention" and
   "doesn't need intervention" impossible with simple thresholds.

### Path to Selective Trace-Only TRACE

The 80.5% S_X probe accuracy demonstrates that residual states can distinguish
reasoning types. The next step is to use residual state classification (rather
than simple R_QK thresholds) for trigger decisions:

- S_X probe predicts: direct_evidence → no intervention
- S_X probe predicts: conflict → conservative/disclose
- S_X probe predicts: evidence_gap → conservative/abstain
- S_X probe predicts: misleading → conservative (not filter)

This would reduce fire rate and preserve direct evidence utility while
maintaining gold-label independence.

## Paper Language

> A trace-only trigger using internal R_QK and confidence signals eliminates
> dependence on gold reasoning-type labels and achieves substantial error
> reduction. However, the current simple threshold trigger is over-conservative
> (92.5% fire rate), causing significant over-abstention on directly answerable
> samples (65% of direct-evidence correct answers lost). This demonstrates that
> gold-label independence is achievable, while also identifying trigger
> calibration — particularly integration of residual-state classification to
> distinguish samples requiring intervention from those that do not — as the
> primary deployment challenge for fully autonomous TRACE.
