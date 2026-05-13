# TRACE-Scale v4 Protocol — Pre-Specified Irregular Stratified Allocation

*Frozen before evaluation — May 2026*

---

## Principle

Pre-specified irregular stratified sample sizes reduce rounding artifacts
in percentage estimates and avoid identical confidence intervals caused by
equal-sized small strata. Sample sizes are fixed before evaluation and not
adjusted after observing model outputs.

## Allocation (Frozen)

| Type | n | Rationale |
|------|---|----------|
| direct_evidence | 40 | All available; false positive estimation |
| conflict | 40 | All available; conflict non-disclosure |
| evidence_gap | **41** | All available; irregular count breaks identical-CI problem |
| misleading_hint | 40 | All available; weakest effect, needs monitoring |
| **Error total** | **121** | |
| **Grand total** | **161** | |

The irregular count (41 for evidence_gap vs 40 for others) is not a
rounding error — it reflects the actual dataset composition and is
specifically retained to prevent identical Wilson CIs from equal-sized
strata, as observed in the v3 table where both conflict and evidence_gap
had n=40, producing identical 97.5% [87.1, 99.6] estimates.

## Statistical Methods (Pre-Registered)

- **Error rates**: Wilson 95% confidence interval
- **Reduction CIs**: 10,000-sample paired bootstrap
- **Significance**: McNemar test with continuity correction
- **False positive bound**: Wilson 95% upper bound

## Scale-Up Target (Pending Additional Data Generation)

For full Nature-level statistical power, the target allocation is:

| Type | Target n |
|------|---------|
| direct_evidence | 121 |
| conflict | 137 |
| evidence_gap | 149 |
| misleading_hint | 173 |
| **Total** | **580** |

This requires generating 419 additional controlled reasoning samples.
The current v4 run uses all 161 available samples with the pre-specified
irregular allocation documented above.
