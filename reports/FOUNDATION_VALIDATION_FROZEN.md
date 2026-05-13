# Foundation Validation — Frozen Baseline

*May 2026*

---

## Repositioned Thesis

> **Transformer reasoning is not attention-only. It proceeds through a staged internal
> computation: QK attention routes evidence, MLPs causally transform evidence into
> reasoning states, residual streams store distributed intermediate representations,
> and logits produce answers.**

ICI is the measurement tool. The mechanism chain is the discovery.

---

## Evidence Matrix (Frozen)

### Dimension 1: QK Routes Evidence

| Metric | GPT-2 (12L) | GPT-2-medium (24L) | Qwen2.5-0.5B (24L) |
|--------|------------|-------------------|---------------------|
| R_QK(direct) | 0.236 | 0.162 | 0.232 |
| R_QK(misleading) | 0.022 | 0.019 | 0.012 |
| Ratio | **10.7×** | **8.5×** | **19.3×** |
| QK reconstruction | 12/12 layers | — | — |

**Finding**: Evidence routing discrimination (8.5–19.3×) holds across GPT-2 and Qwen2.5 architectures.

### Dimension 2: MLP Transforms Evidence

| Metric | GPT-2 | GPT-2-medium |
|--------|-------|-------------|
| MLP ablation |Δlogit| | 114.3 | 56.4 |
| Attention ablation |Δlogit| | 67.4 | 74.5 |
| MLP/Attention ratio | **1.70×** | 0.76× |
| MLP-dominant layers | 10/12 | — |
| MLP patch recovery | 77.1% | — |

**Finding**: MLP is the dominant causal pathway at 12L scale. At 24L, MLP and attention
contributions converge — evidence of distributed computation in deeper models.

### Dimension 3: Residual Stores Reasoning State

| Metric | GPT-2 (50 samples) | GPT-2 (200 samples) |
|--------|-------------------|---------------------|
| Probe accuracy | 64.0% | 80.5% |
| S_X | 0.550 | 0.756 |
| Shuffled labels | 18.8% | 20.4% |
| Permutation p | <0.0001 | <0.0001 |
| Token-count baseline | — | 41.0% (S_X=0.262) |
| Random states | — | 15.0% (S_X=0.000) |

**Finding**: Residual stream encodes reasoning state above all confounds.

### Dimension 4: Mechanism Shifts with Scale

| Model | α(R_QK) | γ(S_X) | Dominant Mode |
|-------|---------|--------|---------------|
| distilgpt2 (6L) | 0.306 | 0.249 | Routing |
| gpt2 (12L) | 0.259 | 0.335 | Transition |
| gpt2-medium (24L) | 0.197 | 0.445 | State encoding |

**Finding**: Systematic shift from routing-dominant to state-encoding-dominant with depth.

### Dimension 5: Visible CoT ≠ Internal Mechanism

| Metric | no-CoT | CoT | Δ |
|--------|--------|-----|---|
| S_X | 0.550 | 0.575 | **+0.025** |
| ICI(direct) | baseline | +0.016 | CoT helps |
| ICI(misleading) | baseline | +0.000 | CoT can't fix |
| Faithful vs Unfaithful | identical | identical | GPT-2 can't discriminate |

**Finding**: CoT amplifies existing internal mechanisms but cannot create mechanisms
where none exist. Visible CoT is an amplifier, not the mechanism itself.

---

## Current Limitations (explicitly stated)

1. **Scale range**: 82M–494M. Larger models (7B+) needed to confirm trend.
2. **Task diversity**: Controlled reasoning tasks. Real-world QA needed.
3. **Two-phase MLP**: Clear at 12L (single-sample per-layer), attenuated in averaged results.
4. **Causal completeness**: Full mediation analysis pending.
5. **Architecture coverage**: GPT-2 + Qwen confirmed. LLaMA, Mistral pending.
6. **Cross-task MLP dominance**: Only direct_evidence tested. Multi-step/conflict pending.

---

## What This Is, And Is Not

**IS**: A mechanistic account of how Transformers convert evidence into answers,
validated through QK reconstruction, MLP causal intervention, residual state probing,
and cross-architecture comparison.

**IS NOT**: A claim that we have fully reverse-engineered Transformer reasoning.
The mechanism chain (QK→MLP→X_l→logits) is a structural skeleton; the detailed
computation within each stage remains for future work.

---

## Next: Where the Mechanism Is Strongest vs Weakest

| Claim | Confidence | Key Gap |
|-------|-----------|---------|
| QK routes evidence | HIGH | Needs LLaMA/Mistral replication |
| MLP dominates at mid-scale | HIGH | Needs scale trend confirmation (7B+) |
| Residual stores reasoning state | HIGH | Needs finer state categories |
| Mechanism shifts with depth | MEDIUM | 3 data points; needs 5+ model sizes |
| Visible CoT ≠ mechanism | HIGH | Needs instruction-tuned models |
| Two-phase MLP universal | LOW | Only GPT-2 12L; needs cross-model replication |

---

*Frozen: Foundation Validation Phase — May 2026.*
