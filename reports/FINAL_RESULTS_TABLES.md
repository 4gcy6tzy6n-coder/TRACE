# Final Results Tables

*Frozen: May 2026 — Nature Manuscript Ready*

---

## Table 1: Mechanism Chain Evidence (GPT-2)

| Stage | Metric | Value |
|-------|--------|-------|
| QK Routing | R_QK(direct) / R_QK(misleading) | 10.7× |
| QK Verification | softmax(QK^T/√d) ≈ A_model | 12/12 layers, tol 10^{-5} |
| MLP Transformation | MLP / Attention \|Δlogit\| ratio | 1.66× |
| MLP Dominance | MLP-dominant layers | 10/12 |
| MLP Patching | Avg recovery ratio | 77.1% |
| Residual State | S_X (5-class probe) | 0.756 |
| Residual Validation | Permutation test p | <0.0001 |
| Residual Control | Token-count baseline S_X | 0.262 |

---

## Table 2: Cross-Model R_QK

| Model | Layers | R_QK(direct) | R_QK(misleading) | Ratio |
|-------|--------|-------------|-----------------|-------|
| distilgpt2 | 6 | 0.255 | 0.025 | 10.2× |
| gpt2 | 12 | 0.236 | 0.022 | 10.7× |
| gpt2-medium | 24 | 0.162 | 0.019 | 8.5× |
| Qwen2.5-0.5B | 24 | 0.232 | 0.012 | 19.3× |

---

## Table 3: Cross-Format MLP Dominance

| Format | MLP > Attention | MLP/Attn Ratio |
|--------|----------------|---------------|
| QA (controlled) | 5/5 | 1.70× |
| FEVER evidence-first | 10/10 | 2.30× |
| FEVER claim-first | 10/10 | 2.96× |
| FEVER QA-style | 10/10 | 2.02× |
| HotpotQA multi-hop | 8/8 | 1.56× |
| HotpotQA single-hop | 8/8 | 1.76× |
| **Total** | **51/51** | — |

---

## Table 4: TRACE-Scale v4 — Primary Error Reduction (Qwen2.5-1.5B)

Pre-specified irregular allocation (n=40, 40, 41, 40). Wilson 95% CI.
10,000-sample paired bootstrap CI. McNemar test.

| Type | n | Raw Errors | Raw [95% CI] | TRACE Errors | TRACE [95% CI] | Reduction [95% CI] | p |
|------|---|-----------|-------------|-------------|---------------|---------------------|---|
| conflict | 40 | 39/40 | 97.5% [87.1, 99.6] | 1/40 | 2.5% [0.4, 12.9] | 97.4% [92.1, 100.0] | <0.0001 |
| evidence_gap | 41 | 40/41 | 97.6% [87.4, 99.6] | 2/41 | 4.9% [1.3, 16.1] | 95.0% [87.5, 100.0] | <0.0001 |
| misleading_hint | 40 | 32/40 | 80.0% [65.2, 89.5] | 30/40 | 75.0% [59.8, 85.8] | 6.3% [-14.3, 24.2] | 0.752 |
| direct_evidence | 40 | 2/40 | 5.0% [1.4, 16.5] | 2/40 | 5.0% [1.4, 16.5] | — | — |
| **Pooled** | **121** | **111/121** | **91.7% [85.5, 95.4]** | **33/121** | **27.3% [20.1, 35.8]** | **70.3% [61.1, 78.8]** | **<0.000001** |

---

## Table 5: Intervention Ablation

| Variant | Errors/121 | Rate [95% CI] | vs TRACE_full | Conclusion |
|---------|-----------|---------------|---------------|------------|
| TRACE_full | 42 | 34.7% [26.8, 43.5] | — | Best |
| no_intervention | 118 | 97.5% [93.0, 99.2] | p<0.0001 | Effect is real |
| mismatched | 78 | 64.5% [55.6, 72.4] | p=0.0013 | Type-matching necessary |
| random | 61 | 50.4% [41.6, 59.2] | p=0.0212 | Mechanism diagnosis matters |

---

## Table 6: Cross-Architecture Error Reduction

| Model | Family | Pooled Reduction [95% CI] | p |
|-------|--------|---------------------------|---|
| Qwen2.5-1.5B | Qwen | 70.3% [61.1, 78.8] | <0.000001 |
| Qwen2.5-3B | Qwen | 57.2% | — |
| LLaMA-3.2-1B-Instruct | LLaMA | 22.1% [12.8, 32.1] | 0.00024 |
| LLaMA-3.2-1B base | LLaMA | ~0% | — |

Consistent pattern across all models: conflict↓, evidence_gap↓, misleading≠, FP=0.

---

## Table 7: Autonomous Trigger Evolution (Qwen2.5-1.5B, n=80)

| Version | Trigger Signals | Error Rate [95% CI] | Fire Rate | Gold-Label Free |
|---------|----------------|---------------------|-----------|-----------------|
| Raw | — | 76.2% [65.9, 84.2] | 0% | — |
| Oracle | reasoning_type | 30.0% [21.1, 40.8] | 75% | ✗ |
| Trace-only | R_QK + conf | 6.2% [2.7, 13.8] | 92% | ✓ |
| **V3.1** | **R_QK + conf + S_X** | **8.8% [4.3, 17.0]** | **71%** | **✓** |
| V3.2 (frontier) | strict S_X gate | 18.8% [11.7, 28.7] | 61% | ✓ |

---

## Table 8: Misleading Mechanism Boundary

| Error Type | Gold Span R_QK | Alternative/Cue R_QK | Pattern | TRACE Strategy | Effective? |
|-----------|---------------|---------------------|---------|---------------|-----------|
| Misleading | 0.005–0.015 | **0.45–0.51** | CUE HIJACKS (13–108×) | Filter | ✗ |
| Conflict | 0.001–0.024 | 0.07–0.13 | Distributed attention | Conservative | ✓ |
