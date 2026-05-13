# Cross-Architecture Gate: CLOSED

*May 2026*

---

## LLaMA-3.2-1B-Instruct: TRACE Error Reduction

| Type | n | Raw | TRACE | Reduction [95% CI] | p |
|------|---|-----|-------|---------------------|---|
| conflict | 31 | 90.3% | 54.8% | **39.3%** [19.2, 58.6] | 0.0055 |
| evidence_gap | 27 | 100% | 77.8% | **22.2%** [7.4, 37.0] | 0.0412 |
| misleading_hint | 22 | 100% | 100% | 0% | 1.0 |
| direct_evidence | 20 | 0% | 0% | — | — |
| **Pooled** | **80** | **96.2%** | **75.0%** | **22.1%** [**12.8, 32.1**] | **0.00024** |

## Cross-Family Comparison

| Model | Family | Pooled Reduction | Conflict | Gap | Misleading | FP |
|-------|--------|-----------------|----------|-----|-----------|-----|
| Qwen2.5-1.5B | Qwen/Llama | **70.3%** [61.1, 78.8] | -100% | -100% | -22.2% | 0% |
| Qwen2.5-3B | Qwen/Llama | **57.2%** | -71.4% | -100% | 0% | 0% |
| LLaMA-3.2-1B-Instruct | **LLaMA** | **22.1%** [12.8, 32.1] | -39.3% | -22.2% | 0% | 0% |
| LLaMA-3.2-1B base | LLaMA | ~0% | ~0% | ~0% | 0% | 0% |

## Consistent Pattern Across ALL Families

1. **Conflict: reduced** — all three >1B models show significant reduction
2. **Evidence gap: reduced** — all three models show reduction
3. **Misleading: resistant** — zero or non-significant in ALL models
4. **Direct evidence: 0% false positives** — ALL models

## The LLaMA Base/Instruct Distinction

LLaMA-3.2-1B **base** (no instruction training): ~0% effect. Conservative prompting
ignored; model treats "say Cannot determine" as continuation text.

LLaMA-3.2-1B **Instruct** (instruction-tuned): 22.1% reduction, p=0.00024.
Same TRACE diagnosis, same intervention prompts, same error type matching —
the difference is the model's ability to execute the corrective instruction.

This reveals an important boundary: TRACE has two layers.
- **Layer 1 (Audit/Diagnosis)**: identifies internal mechanism weakness
- **Layer 2 (Intervention)**: requires model instruction-following capability to execute correction

## Nature-Level Status

| Gate | Status |
|------|--------|
| Cross-model replication (GPT-2) | ✓ |
| Cross-model replication (Qwen2.5) | ✓ |
| **Cross-family replication (LLaMA)** | **✓** |
| Cross-format replication (FEVER + HotpotQA) | ✓ |
| Prompt engineering confound | ✓ |
| Statistical rigor | ✓ |
| Intervention ablation | ✓ |
| Misleading mechanism boundary | ✓ |
