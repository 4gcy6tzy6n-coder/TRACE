# Nature Manuscript Assembly — Ready

*May 2026*

---

## Status: All Gates Closed

| Gate | Status |
|------|--------|
| Mechanism discovery (QK→MLP→X_l→logits) | ✓ |
| Cross-model validation (GPT-2 + Qwen2.5) | ✓ |
| Cross-format validation (QA + 3 FEVER + HotpotQA) | ✓ |
| Real-task validation (FEVER-100 + HotpotQA-60) | ✓ |
| TRACE audit (selective risk detection) | ✓ |
| TRACE intervention (mechanism-matched) | ✓ |
| >1B error reduction (1 model) | ✓ |
| **>1B error reduction (≥2 models)** | **✓** |

## Core Thesis

> We identify a staged evidence-to-answer mechanism in Transformers
> (QK→MLP→X_l→logits) and convert it into TRACE, a mechanism-grounded
> auditing and intervention framework that reduces black-box reasoning
> failures by 57–75% across two >1B-parameter models without increasing
> false positives on direct-evidence cases.

## Five Contributions

1. **Mechanism chain**: QK routes evidence, MLP transforms it, residual stores state, logits produce answers
2. **MLP dominance**: MLP is the primary causal pathway (1.43–2.96× over attention, 54/54 cross-format)
3. **Residual state**: Residual streams encode reasoning states (S_X = 0.50–0.99, all p < 0.0001)
4. **Cross-validation**: Mechanism holds across architectures (GPT-2, Qwen2.5) and task formats (QA, FEVER, HotpotQA)
5. **TRACE**: Mechanism-grounded intervention reduces black-box failures by 57–75% on >1B models

## Key Result (TRACE-Scale)

| Model | Total Error Reduction | Conflict | Evidence Gap | Misleading | False Positives |
|-------|----------------------|----------|-------------|-----------|-----------------|
| Qwen2.5-1.5B | **-75.0%** | -100% | -100% | -22.2% | 0% |
| Qwen2.5-3B | **-57.2%** | -71.4% | -100% | 0% | 0% |

## Manuscript Assembly Order

1. Abstract (Nature-style, ≤150 words)
2. Figure plan (6 main, 2 supplementary)
3. Introduction (§1 — drafted)
4. Results (§4 — drafted, needs TRACE-Scale update)
5. Methods (§3 — drafted)
6. Discussion (§5 — needs writing)
7. Related Work (§2 — needs writing)
8. Final pass for Nature formatting

## Ship Files

- `TRACE_SCALE_GATE_CLOSED.md` — final error reduction results
- `CLAIM_BOUNDARY.md` — what to claim and what not to claim
- `FIGURE_PLAN.md` — updated with TRACE-Scale figures
- `draft/01-03` — Introduction, Results, Methods (drafted)
