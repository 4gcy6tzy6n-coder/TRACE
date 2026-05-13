# FEVER-Style Fact Verification Validation

## Result Summary

| Metric | QA Format (controlled) | FEVER Format | Interpretation |
|--------|----------------------|-------------|----------------|
| R_QK ratio (direct/misleading) | **10.6×** | 1.1× (SUPPORTS/NEI) | Format-dependent |
| MLP > Attention | **5/5** (100%) | **5/5** (100%) | Format-robust |
| S_X (probe accuracy) | **0.50–0.76** (p<0.0001) | 0.10 (p=0.84) | Format-dependent |

## Key Findings

### 1. MLP Dominance Is Format-Robust

MLP ablation caused 2–6× larger logit changes than attention ablation in all 5
tested FEVER-style samples. This is the strongest cross-format signal and
supports the claim that MLP is the primary evidence-to-answer transformation
pathway regardless of task framing.

### 2. R_QK Separation Is Format-Dependent

In QA format, direct/misleading R_QK separation is 10.6×. In FEVER format,
SUPPORTS/REFUTES/NEI R_QK values are nearly identical (0.015, 0.017, 0.013).
This is expected: in the FEVER format, evidence is presented as a single block
before the claim, so attention distributes similarly regardless of verdict type.
The QA format (question after evidence) creates stronger routing differentiation.

### 3. S_X Requires Adequate Data

With only 20 samples (7 SUPPORTS, 11 REFUTES, 2 NEI), the 3-class probe
achieves 40% accuracy (S_X=0.10, p=0.84). This is not significant. Larger,
balanced datasets are needed for reliable verdict-type probing.

## Limitations

- Data: 20 restructured samples, not real FEVER dataset
- Label imbalance: 2 NEI samples (insufficient for CV)
- Evidence format: single evidence block, not multi-document FEVER format
- No claim-evidence pair structure (real FEVER has structured claim-evidence pairs)

## Next Steps

1. Download real FEVER shared task data for 100+ balanced samples
2. Use structured claim-evidence pairs with annotated evidence spans
3. Test with HotpotQA for multi-hop validation
4. Consider that format-dependent R_QK may itself be a finding:
   "Evidence routing strength depends on task framing, but MLP dominance does not"
