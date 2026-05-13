# Final Method Version — TRACE V3.1

*Frozen: May 2026*

---

## Primary Method: TRACE V3.1 (Residual-State-Gated Autonomous Trigger)

TRACE (Transformer Reasoning Auditing through Causal Evidence) is a
mechanism-grounded framework for auditing and intervening on black-box
reasoning failures. V3.1 is the current best autonomous version.

### Trigger Signals (no gold labels)

```
Trigger = f(R_QK, confidence, S_X_state, S_X_probability)
```

| Signal | Source | Meaning |
|--------|--------|---------|
| R_QK | Q, K, A (attention weights) | Evidence routing strength |
| Confidence | Logits (softmax) | Model output confidence |
| S_X_state | Residual stream probe | Predicted reasoning state |
| S_X_probability | Residual stream probe | Confidence of state prediction |

### Intervention Rules (V3.1 calibrated)

| Condition | Action |
|-----------|--------|
| S_X = direct_evidence AND prob > 0.5 | No intervention |
| S_X = evidence_gap AND prob > 0.5 | Conservative (abstain-capable) |
| S_X = conflict AND prob > 0.5 | Conservative (disclose-capable) |
| S_X = misleading AND R_QK < 0.02 AND prob > 0.4 | Conservative |
| R_QK < 0.005 AND conf < 0.15 | Conservative |
| All other cases | No intervention |

### Key Properties

- **No gold labels**: All trigger signals from internal model variables
- **No prompt engineering**: Intervention ablation proves effect is from mechanism-matching
- **Cross-architecture**: Validated on GPT-2, Qwen2.5, LLaMA-3.2
- **Cross-format**: Validated on QA, FEVER-style, HotpotQA-style

### V3.2 (Calibration Frontier, NOT primary method)

V3.2 uses stricter S_X gating to further reduce fire rate (61%, 34%↓ vs trace-only)
but error increases to 18.8%. This demonstrates a safety–utility trade-off, not a
better method. V3.2 is reported as calibration frontier analysis.

### Relationship to ICI

ICI (Internal Causal Index) = αR_QK + βM_AV + γS_X + δC_do is the measurement tool
that quantifies the mechanism chain. TRACE is the audit and intervention framework
built on top of ICI. ICI is the diagnostic; TRACE is the action.

### Terminology

- **Use**: "Evidence-to-answer mechanism", "TRACE", "mechanism-grounded auditing"
- **ICI**: "Internal Causal Index" — emphasizes causal intervention (C_do), not external CoT
