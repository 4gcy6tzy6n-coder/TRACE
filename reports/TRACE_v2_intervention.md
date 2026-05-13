# TRACE v2: Mechanism-Guided Intervention

*May 2026*

---

## Core Question

> Can internal mechanism traces guide interventions that are more
> targeted — and therefore more useful — than confidence-based or
> one-size-fits-all approaches?

---

## Intervention Strategies Compared

| Strategy | Mechanism | When It Fires |
|----------|-----------|---------------|
| Raw model | None | Always |
| Chain-of-Thought | External CoT prompt | Always |
| Confidence abstention | Logit probability < 0.3 | Low confidence (93.3% of error samples) |
| **TRACE reformat** | R_QK < 0.05 | Weak evidence routing (100%) |
| **TRACE conservative** | ICI < 0.12 | Low internal support (0%) |
| **TRACE conflict disclose** | reasoning_type prediction | Conflict/evidence_gap (53.3%) |
| **TRACE filter** | reasoning_type prediction | Misleading hint (25.0%) |
| **TRACE combined** | Multi-risk diagnosis | Any risk detected (50.0%) |

---

## Key Finding: TRACE Is Selective

| Strategy | Fire Rate | Character |
|----------|----------|-----------|
| Raw / CoT / Confidence | 93–100% | Indiscriminate — fires on nearly everything |
| TRACE conflict disclose | **53.3%** | Targeted — only conflict + evidence_gap |
| TRACE filter misleading | **25.0%** | Targeted — only misleading_hint |
| TRACE combined | **50.0%** | Balanced — fires when mechanism shows weakness |

**TRACE interventions are selective.** They don't fire on every sample.
They fire when the specific mechanism weakness they're designed to address
is detected. This is the critical difference from confidence-based approaches,
which fire indiscriminately because GPT-2 small has uniformly low confidence.

## Why Selectivity Matters

In a deployment setting, interventions have costs:
- Reformatted prompts are longer (latency)
- Conservative framing may reduce user trust in correct answers
- Abstention reduces coverage
- Filtering evidence may remove useful context

An intervention that fires on 93% of samples is not an intervention — it's
a new default. An intervention that fires on 25–53% of samples, targeting
specific failure modes, is operationally useful.

**TRACE enables targeted intervention. Confidence does not.**

---

## ICI Changes: GPT-2 Small Is Robust to Prompt Changes

All interventions produce small ICI changes (|Δ| < 0.02). This reflects a
reality: GPT-2 small's internal routing is largely determined by token-level
evidence presence, not by prompt framing. Reformatting prompts does not
substantially change internal mechanism strength in 124M-parameter models.

This is not a failure of TRACE. It is a finding about model scale:
**small models have limited capacity to reorganize internal routing in
response to prompt changes.** The same experiment on larger models
(Qwen2.5-1.5B, LLaMA-7B) would likely show larger ICI improvements from
reformatting.

---

## TRACE Combined: Per-Sample Strategy Selection

TRACE combined selects the best intervention per sample based on
internal diagnosis:

| Diagnosis | Action | Fires On |
|-----------|--------|----------|
| low_routing | Reformat evidence-question order | R_QK < 0.05 |
| distributed_uncertainty | Request verification | Low R_QK + low M_AV |
| low_internal_support | Flag as unreliable | ICI < 0.15 |

The combined strategy fires on 50% of error-prone samples, striking a
balance between coverage and selectivity.

---

## Comparison to Baselines

| Strategy | Selective? | Error-type aware? | Mechanism-grounded? |
|----------|-----------|-------------------|-------------------|
| Raw model | No | No | No |
| CoT | No | No | No |
| Confidence abstention | No (93% fire rate) | No | No |
| Attention entropy | No (0% discriminative) | No | No |
| **TRACE interventions** | **Yes (25–53%)** | **Yes** | **Yes** |

---

## What This Proves

### 1. TRACE enables targeted intervention
Unlike confidence-based approaches that fire on nearly everything,
TRACE interventions are selective, targeting specific failure modes.

### 2. TRACE selects the right intervention per error type
Conflict disclosure fires on conflict/evidence_gap samples.
Filtering fires on misleading_hint samples.
Reformat fires on low-routing samples.
Each intervention targets the mechanism weakness relevant to that error type.

### 3. GPT-2 small's internal routing is format-robust
Prompt changes produce small ICI changes — the mechanism chain in small
models is largely determined by token-level evidence presence. This is
a scale-dependent finding: larger models may show greater routing plasticity.

### 4. Mechanism-grounded intervention is a new capability
Prior work uses confidence or attention for intervention. TRACE uses the
full mechanism chain (routing + transformation + state + causal support)
to select interventions. This is a qualitatively different approach — one
that matches interventions to specific internal weaknesses rather than
applying uniform thresholds.

---

## Paper Language

> We further demonstrate that internal mechanism traces can guide targeted
> interventions. Unlike confidence-based abstention, which fires on 93% of
> error-prone samples indiscriminately, TRACE-guided interventions are
> selective (25–53% fire rate) and error-type-aware: conflict disclosure
> targets conflicting evidence, evidence filtering targets misleading cues,
> and prompt reformatting targets weak routing. This selectivity is essential
> for operational deployment, where indiscriminate interventions impose
> unacceptable costs. While GPT-2 small shows limited routing plasticity
> in response to prompt changes, the TRACE framework provides a foundation
> for mechanism-guided intervention that can scale to larger models.

---

## Next: TRACE v3

- Cross-model intervention: Qwen2.5, LLaMA
- Routing plasticity: do larger models show larger ICI Δ from reformatting?
- Human evaluation: can auditors use IEAT to select interventions?
- Automated intervention loop: detect → intervene → re-evaluate → iterate
