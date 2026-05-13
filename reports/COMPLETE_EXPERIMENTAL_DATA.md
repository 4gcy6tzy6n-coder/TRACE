# Complete Experimental Data Summary

*May 2026 — All versions, all models, all experiments*

---

## 一、机制发现 (Mechanism Discovery, v0.1–v0.3)

### 1.1 QK Evidence Routing (R_QK) — GPT-2

| Reasoning Type | R_QK |
|---------------|------|
| direct_evidence | 0.230 |
| evidence_gap | 0.018 |
| multi_step | 0.015 |
| conflict | 0.030 |
| misleading_hint | 0.011 |

Ratio: direct/misleading = **20.9×** (GPT-2), **10.7×** (pooled)

### 1.2 MLP vs Attention Ablation — GPT-2

| Component | \|Δlogit\| |
|-----------|-----------|
| Attention output | 59.3 |
| MLP output | **98.1** |
| MLP/Attention ratio | **1.66×** |
| MLP-dominant layers | **10/12** |

### 1.3 Residual State Probe (S_X) — GPT-2

| Probe Setup | Accuracy | S_X |
|------------|----------|-----|
| Real residual (200 samples) | 80.5% ± 4.0% | **0.756** |
| Shuffled labels (100 permutations) | 20.4% ± 3.2% | — |
| Random hidden states | 15.0% | 0.000 |
| Token-count baseline | 41.0% | 0.262 |
| Permutation test | p < **0.0001** | — |

### 1.4 MLP Activation Patching — GPT-2

| Metric | Value |
|--------|-------|
| Avg MLP patch recovery | **77.1%** |
| >90% recovery | 3/5 pairs |

### 1.5 Two-Phase MLP — GPT-2

| Layer Group | Sign | Interpretation |
|------------|------|----------------|
| Early (L0-2) | Negative (logit↑ after ablation) | Suppression |
| Middle (L3-7) | Positive (logit↓ after ablation) | Construction |
| Late (L8-11) | Negative (logit↑ after ablation) | Suppression |

---

## 二、跨模型验证 (Cross-Model Validation, v0.4)

### 2.1 R_QK by Model

| Model | Layers | R_QK(direct) | R_QK(misleading) | Ratio |
|-------|--------|-------------|-----------------|-------|
| distilgpt2 | 6 | 0.255 | 0.025 | **10.2×** |
| gpt2 | 12 | 0.236 | 0.022 | **10.7×** |
| gpt2-medium | 24 | 0.162 | 0.019 | **8.5×** |
| Qwen2.5-0.5B | 24 | 0.232 | 0.012 | **19.3×** |

### 2.2 S_X by Model

| Model | Layers | S_X |
|-------|--------|-----|
| distilgpt2 | 6 | 0.583 |
| gpt2 | 12 | 0.542 |
| gpt2-medium | 24 | **0.708** |
| Qwen2.5-0.5B | 24 | 0.70 (permutation p<0.0001) |

### 2.3 MLP Dominance by Model

| Model | MLP/Attn Ratio | MLP Dominant? |
|-------|---------------|---------------|
| gpt2 (12L) | **1.70×** | ✓ |
| Qwen2.5-0.5B (24L) | **1.43×** | ✓ |
| gpt2-medium (24L) | 0.76× | ✗ (converges) |

---

## 三、跨格式验证 (Cross-Format, v0.4)

### 3.1 FEVER-100: 3 Format Variants

| Format | MLP>Attn | MLP/Attn | S_X | p |
|--------|---------|---------|-----|---|
| evidence-first | **10/10** | 2.30× | 0.925 | <0.0001 |
| claim-first | **10/10** | 2.96× | 0.910 | <0.0001 |
| qa-style | **10/10** | 2.02× | 0.985 | <0.0001 |

**MLP > Attention: 30/30 samples across 3 FEVER formats.**

### 3.2 HotpotQA-60: Multi-hop vs Single-hop

| Type | MLP>Attn | MLP/Attn | S_X (multi vs single) |
|------|---------|---------|----------------------|
| multi-hop | **8/8** | 1.56× | — |
| single-hop | **8/8** | 1.76× | — |
| Multi vs Single (probe) | — | — | **0.922**, p<0.0001 |

**MLP > Attention: 16/16 HotpotQA samples.**

### 3.3 Cross-Format Total

| Format | MLP>Attn Samples |
|--------|-----------------|
| QA (controlled) | 5/5 |
| FEVER (3 variants) | 30/30 |
| HotpotQA (2 variants) | 16/16 |
| **Total** | **54/54** |

---

## 四、TRACE 审计 (Risk Detection, v1)

### 4.1 Risk Detection by Type (GPT-2)

| Reasoning Type | % Flagged as Risky |
|---------------|-------------------|
| direct_evidence | **0.0%** |
| conflict | 21.1% |
| evidence_gap | 29.2% |
| misleading_hint | **88.9%** |
| multi_step | **94.4%** |

### 4.2 Internal Trace vs External Signals

| Signal | Precision | Recall | F1 |
|--------|----------|--------|-----|
| **TRACE (internal)** | **0.455** | 0.541 | 0.494 |
| Logit confidence | 0.370 | 1.000 | 0.540 |
| Attention entropy | 0.000 | 0.000 | 0.000 |

---

## 五、TRACE 干预 (Error Reduction, v2–v4)

### 5.1 Primary Result: Qwen2.5-1.5B (n=121, v4 irregular allocation)

| Type | n | Raw Errors | Raw [95% CI] | TRACE Errors | TRACE [95% CI] | Rel. Reduction [95% CI] | p |
|------|---|-----------|-------------|-------------|---------------|------------------------|---|---|
| conflict | 40 | 39/40 | 97.5% [87.1, 99.6] | 1/40 | 2.5% [0.4, 12.9] | **97.4%** [92.1, 100.0] | <0.0001*** |
| evidence_gap | 41 | 40/41 | 97.6% [87.4, 99.6] | 2/41 | 4.9% [1.3, 16.1] | **95.0%** [87.5, 100.0] | <0.0001*** |
| misleading_hint | 40 | 32/40 | 80.0% [65.2, 89.5] | 30/40 | 75.0% [59.8, 85.8] | 6.3% [-14.3, 24.2] | 0.752 |
| direct_evidence | 40 | 2/40 | 5.0% [1.4, 16.5] | 2/40 | 5.0% [1.4, 16.5] | — | — |
| **Pooled** | **121** | **111/121** | **91.7% [85.5, 95.4]** | **33/121** | **27.3% [20.1, 35.8]** | **70.3%** [**61.1, 78.8**] | **<0.000001***** |

### 5.2 TRACE vs Baselines (Qwen2.5-1.5B)

| Strategy | Error Rate | Safe Output | Fire Rate | Character |
|----------|-----------|------------|-----------|-----------|
| Raw model | 91.7% | 0.8% | 0% | Baseline |
| Chain-of-Thought | 91.7% | 2.5% | 100% | No improvement |
| Confidence abstention | 45.5% | 39.7% | 32.2% | Indiscriminate |
| Attention entropy | 10.7% | 71.9% | 78.5% | Over-abstains |
| **TRACE selective** | **27.3%** | **57.0%** | **73.6%** | **Mechanism-matched** |

---

## 六、干预消融 (Intervention Ablation, v4)

### 6.1 Qwen2.5-1.5B (n=121)

| Variant | Errors | Rate [95% CI] | vs TRACE_full |
|---------|--------|---------------|---------------|
| TRACE_full | 42/121 | 34.7% [26.8, 43.5] | — |
| no_intervention | 118/121 | 97.5% [93.0, 99.2] | p<0.0001 *** |
| mismatched | 78/121 | 64.5% [55.6, 72.4] | p=0.0013 ** |
| random | 61/121 | 50.4% [41.6, 59.2] | p=0.0212 * |

**结论**：
- TRACE效果不是prompt engineering（mismatched 1.9× worse）
- 错误类型匹配是必要的（mismatched显著更差）
- 机制诊断有真实信号（random显著更差）

---

## 七、跨架构复制 (Cross-Architecture, v4)

### 7.1 LLaMA-3.2-1B-Instruct (n=80 error + 20 direct)

| Type | n | Raw | TRACE | Reduction [95% CI] | p |
|------|---|-----|-------|---------------------|---|
| conflict | 31 | 90.3% | 54.8% | **39.3%** [19.2, 58.6] | 0.0055 |
| evidence_gap | 27 | 100% | 77.8% | **22.2%** [7.4, 37.0] | 0.0412 |
| misleading_hint | 22 | 100% | 100% | 0% | 1.0 |
| direct_evidence | 20 | 0% | 0% | — | — |
| **Pooled** | **80** | **96.2%** | **75.0%** | **22.1%** [**12.8, 32.1**] | **0.00024** |

### 7.2 LLaMA-3.2-1B Base (n=80 error + 20 direct)

| Type | Raw → TRACE | 说明 |
|------|------------|------|
| 全部类型 | ~0% reduction | Base model 不执行指令式干预 |

### 7.3 跨家族对比

| Model | Family | Pooled Reduction | Conflict | Gap | Misleading | FP |
|-------|--------|-----------------|----------|-----|-----------|-----|
| Qwen2.5-1.5B | Qwen/Llama | **70.3%** [61.1, 78.8] | -100% | -100% | -22.2% | 0% |
| Qwen2.5-3B | Qwen/Llama | **57.2%** | -71.4% | -100% | 0% | 0% |
| LLaMA-3.2-1B-Instruct | LLaMA | **22.1%** [12.8, 32.1] | -39.3% | -22.2% | 0% | 0% |

**一致模式**：conflict↓, gap↓, misleading≠, FP=0 — 跨三个模型、两个架构家族。

---

## 八、Misleading 机制分析 (Mechanism Boundary)

### 8.1 Dual-Span R_QK: Misleading vs Conflict (Qwen2.5-1.5B)

| Error Type | Gold Span R_QK | Alternative/Cue R_QK | Pattern |
|-----------|---------------|---------------------|---------|
| Misleading | 0.005–0.015 | **0.45–0.51** | **CUE HIJACKS** (13–108×) |
| Conflict | 0.001–0.024 | 0.07–0.13 | Distributed attention |

**结论**：Misleading 证据劫持 QK routing，cue 词获得 13–108× 更高的 attention。Filter 无法重定向 routing。Conflict 是 attention 分散，可以触发披露。

---

## 九、Scale-Aware 权重校准 (v0.5)

| Model | Layers | α(R_QK) | β(M_AV) | γ(S_X) | 主导模式 |
|-------|--------|---------|---------|--------|---------|
| distilgpt2 | 6 | **0.306** | 0.204 | 0.249 | Routing-dominant |
| gpt2 | 12 | 0.259 | 0.172 | 0.335 | Transition |
| gpt2-medium | 24 | 0.197 | 0.132 | **0.445** | State-dominant |

**结论**：α↓ (0.306→0.197), γ↑ (0.249→0.445) — 随深度从 routing-dominant 转向 state-encoding-dominant。

---

## 十、CoT 效应 (v0.3)

| Metric | no-CoT | CoT | Δ |
|--------|--------|-----|---|
| S_X | 0.550 | 0.575 | **+0.025** |
| ICI(direct) | baseline | +0.016 | CoT 帮助 |
| ICI(misleading) | baseline | +0.000 | CoT 无法修复 |
| Faithful vs Unfaithful | identical | identical | GPT-2 无法区分 |

---

## 十一、模型清单

| Model | Params | Layers | Architecture | 状态 |
|-------|--------|--------|-------------|------|
| distilgpt2 | 82M | 6 | GPT-2 | ✓ |
| gpt2 | 124M | 12 | GPT-2 | ✓ |
| gpt2-medium | 355M | 24 | GPT-2 | ✓ |
| pythia-70m | 70M | 6 | NeoX | fp16 NaN |
| pythia-160m | 160M | 12 | NeoX | fp16 NaN |
| Qwen2.5-0.5B | 494M | 24 | Qwen/Llama | ✓ |
| Qwen2.5-1.5B | 1.5B | 28 | Qwen/Llama | ✓ |
| Qwen2.5-3B | 3B | 36 | Qwen/Llama | ✓ |
| LLaMA-3.2-1B base | 1.1B | 16 | LLaMA | ✓ (diagnosis only) |
| LLaMA-3.2-1B-Instruct | 1.1B | 16 | LLaMA | ✓ |
| TinyLlama-1.1B-Chat | 1.1B | 22 | LLaMA | Chat format mismatch |

---

## 十二、数据清单

| Dataset | Samples | Types |
|---------|---------|-------|
| toy_reasoning.jsonl | 200 | direct, conflict, gap, misleading, multi-step |
| fever_100.jsonl | 100 | SUPPORTS/REFUTES/NEI (40/40/20) |
| hotpot_style.jsonl | 60 | multi-hop + single-hop |
| corrupted_pairs.jsonl | 15 | clean/corrupted pairs |
| faithful_unfaithful_cot.jsonl | 10 | faithful + unfaithful CoT pairs |

---

## 十三、统计方法清单

| Method | Purpose |
|--------|---------|
| Wilson 95% CI | Error rate estimation (handles 0% and 100%) |
| 10,000-sample paired bootstrap CI | Reduction estimate uncertainty |
| McNemar test (continuity correction) | Paired binary outcome significance |
| Permutation test (100 shuffles) | Probe significance |
| Pre-specified irregular stratified allocation | Avoid identical CIs from equal-sized strata |
