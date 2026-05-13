# Misleading Failure Mechanism Analysis

*May 2026*

---

## Core Finding

**Misleading evidence hijacks QK routing.** The model overwhelmingly attends to
misleading cues (R_QK ≈ 0.45–0.51) rather than corrective evidence
(R_QK ≈ 0.005–0.015). This is fundamentally different from conflict,
where attention distributes across both evidence spans.

## Dual-Span R_QK Analysis (Qwen2.5-1.5B)

### Misleading Samples (n=15)

| Sample | R_QK(gold evidence) | R_QK(misleading cue) | Dominant |
|--------|--------------------|---------------------|----------|
| case_031 | 0.005 | 0.497 | CUE (108×) |
| case_032 | 0.011 | 0.458 | CUE (41×) |
| case_033 | 0.007 | 0.450 | CUE (68×) |
| case_035 | 0.005 | 0.063 | CUE (13×) |
| case_036 | 0.005 | 0.452 | CUE (98×) |

**Result**: CUE WINS in 11/15 samples. Misleading evidence dominates QK routing
by 13–108× over corrective evidence.

### Conflict Samples (n=10)

| Sample | R_QK(gold span) | R_QK(alternative) | Pattern |
|--------|----------------|-------------------|---------|
| case_011 | 0.014 | 0.094 | ALT preferred |
| case_012 | 0.024 | 0.087 | ALT preferred |
| case_013 | 0.002 | 0.105 | ALT strongly preferred |

**Result**: Both spans get moderate attention, but the alternative span
consistently dominates the gold span. Attention is distributed, not hijacked.

## Why TRACE Filter Fails for Misleading

Filtering misleading cue words ("claims", "ads say", "influencers") from
the prompt does NOT remove the misleading evidence tokens themselves.
The misleading narrative — "supplement boosts energy in 90% of users" —
is carried by content words that remain after filtering. The model continues
to route attention to these content words, which support the misleading
interpretation.

R_QK to gold evidence remains low (0.005–0.015) even after filtering,
because the gold evidence ("no significant effect over placebo") is a
small span embedded in text that the model treats as primarily misleading.

## Why TRACE Conservative Works for Conflict

Conservative prompting ("if evidence is conflicting, say Cannot determine")
works because the model attends to BOTH evidence spans. When prompted to
check for conflict, the model can recognize the distributed attention
pattern and disclose the conflict. The key difference: conflict attention
is distributed, misleading attention is hijacked.

## Mechanism Boundary

| Error Type | QK Pattern | TRACE Strategy | Works? | Why |
|-----------|-----------|---------------|--------|-----|
| Conflict | Distributed across spans | Conservative/abstain | ✓ | Model sees both sides |
| Evidence gap | No strong routing | Conservative/abstain | ✓ | Model recognizes absence |
| **Misleading** | **Hijacked to misleading cue** | **Filter** | **✗** | **Filter doesn't redirect routing** |

## Implication

Prompt-level intervention cannot repair hijacked QK routing. The misleading
evidence tokens themselves carry semantic content that the model treats as
credible evidence. To fix misleading-driven errors, we would need:
1. External source reliability signaling
2. Counterfactual evidence presentation
3. Multi-pass routing with explicit debiasing
4. Training-time resistance to misleading evidence patterns

This is a mechanism boundary, not a TRACE failure. TRACE correctly diagnoses
misleading samples as high-risk (88.9% flagged in v1); the limitation is in
the intervention, not the diagnosis.
