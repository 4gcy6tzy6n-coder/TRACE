# Evidence Routing and Distributed Answer Computation in Transformers
# A Mechanistic Account of Internal Reasoning

*Research repositioning document — May 2026. Updated: cross-model validation.*

---

## Core Thesis (Refined)

**Transformer reasoning proceeds through a staged internal computation: QK attention
robustly routes evidence across architectures, residual streams store distributed
reasoning states, and answer computation shifts from MLP-dominant transformation
at intermediate depth to distributed pathways in deeper models.**

Cross-model validation (GPT-2 12L, GPT-2-medium 24L, Qwen2.5-0.5B 24L):
- QK routing: universal (3/3 models, 7.6×–17.4× ratio)
- Residual state encoding: universal (3/3 models, S_X=0.50–0.70, all p<0.0001)
- MLP dominance: depth-dependent (2/3 models; converges with attention at 24L)

---

## 1. The Problem

The dominant paradigm for interpreting Transformer reasoning focuses on attention weights
and chain-of-thought prompting. But attention visualizations conflate routing with
computation, and visible CoT may not reflect internal mechanisms.

We ask the more fundamental question:

> **How do Transformers internally convert evidence into answers?**

This question matters beyond any single interpretability method. If we understand the
internal mechanism, we can diagnose failures, detect unfaithful reasoning, and design
better architectures — regardless of whether external CoT is available.

---

## 2. The Mechanism Chain

Through systematic decomposition of Transformer internal variables
($Q, K, V, A, W_O, W_U, X_l$, MLP), we identify a four-stage mechanism:

```
                    QK ATTENTION
                   routes evidence
                        │
                        ▼
                    MLP LAYERS
            causally transform evidence
              into reasoning states
                        │
                        ▼
                  RESIDUAL STREAM
           stores distributed intermediate
                reasoning states
                        │
                        ▼
                     LOGITS
                 produce answer
```

### Stage 1: QK Routes Evidence

QK attention weights identify evidence-relevant tokens. Evidence routing strength
($R_{QK}$, attention mass from answer to evidence positions) is 8.5×–19.3× higher
for direct-evidence samples than misleading-hint samples across GPT-2 and Qwen2.5
architectures. QK reconstruction verified: $\text{softmax}(QK^T/\sqrt{d})$ matches
model attention weights in all 12 GPT-2 layers within $10^{-5}$ tolerance.

**Finding**: QK attention routes evidence, but routing alone does not produce the answer.

### Stage 2: MLP Transforms Evidence into Reasoning States

MLP layers — not attention head projections — are the dominant causal pathway for
evidence-to-answer transformation. MLP output ablation causes 1.66× larger logit
changes than attention output ablation (|Δlogit| = 98.1 vs 59.3). MLP activation
patching from clean to corrupted samples recovers correct answers with 77.1% average
recovery ratio. Layer-wise analysis reveals a two-phase MLP computation: early and
late MLPs suppress premature answer output, while middle MLPs construct the
evidence-grounded signal. 10 of 12 layers are MLP-dominant by causal effect size.

**Finding**: Attention finds evidence; MLP computes answers.

### Stage 3: Residual Stream Stores Distributed Reasoning States

Linear probes trained on residual stream activations classify reasoning types
(direct evidence, conflict, evidence gap, misleading hint, multi-step) with
80.5% accuracy (5-class random = 20%, $S_X = 0.756$). Permutation test:
$p < 0.0001$ (100 shuffles). Random hidden states: $S_X = 0.000$.
Token-count baseline: 41.0% accuracy. Residual-state separability increases
with model depth: $S_X$ rises from 0.583 (6L) to 0.708 (24L).

**Finding**: Residual streams serve as distributed workspaces encoding
intermediate reasoning states, with encoding capacity increasing with model depth.

### Stage 4: Mechanism Shifts with Scale

As models deepen, internal reasoning shifts systematically from QK-routing-dominant
to residual-state-dominant. Scale-aware weight calibration reveals:
- 6-layer model: $\alpha(R_{QK}) = 0.306 > \gamma(S_X) = 0.249$
- 24-layer model: $\alpha(R_{QK}) = 0.197 < \gamma(S_X) = 0.445$

MLP-mediated evidence processing peaks at intermediate scales (12 layers: 42.8% MLP
fraction) and distributes across pathways at larger scales.

**Finding**: Internal reasoning is not one mechanism but a collection of mechanisms
that scale differently — a finding hidden by fixed-weight composite metrics.

---

## 3. Evidence Summary

| Claim | Evidence | Version |
|-------|----------|---------|
| QK routes evidence, not noise | $R_{QK}$(direct) : $R_{QK}$(misleading) = 8.5×–19.3× across architectures | v0.4 |
| QK reconstruction verified | $\text{softmax}(QK^T/\sqrt{d}) \approx A_{model}$ (all 12 layers, tol $10^{-5}$) | v0.2 |
| MLP is dominant causal pathway | MLP |Δlogit| = 98.1 vs attention = 59.3; 10/12 layers MLP-dominant | v0.6 |
| MLP patching recovers answer | 77.1% average recovery; >90% for 3/5 pairs | v0.6 |
| Two-phase MLP: suppress/construct | Early+late MLP suppress (Δ<0), middle MLP construct (Δ>0) | v0.6 |
| Residual encodes reasoning state | $S_X = 0.756$, permutation $p < 0.0001$, token-count baseline 0.262 | v0.3 |
| $S_X$ increases with depth | 0.583 (6L) → 0.708 (24L) | v0.4 |
| Weight shift: routing → state | $\alpha$↓ (0.306→0.197), $\gamma$↑ (0.249→0.445) with depth | v0.5 |
| MLP pathway > attention pathway | MLP fraction 13-43% vs attention fraction 4-7% | v0.5 |
| CoT boosts $S_X$ but not misleading | $S_X$ +25% with CoT; misleading ICI unchanged | v0.3 |
| Strict patching controls pass | Correct > random, unrelated, same-type-wrong; evidence-position specific | v0.3 |

---

## 4. Why This Matters Beyond Interpretability

### 4.1 For Model Diagnosis

If misleading prompts leave detectable internal traces (low $R_{QK}$, suppressed MLP
construction, reduced $S_X$), we can detect unfaithful reasoning without needing
external CoT — directly from internal activations.

### 4.2 For Architecture Design

If MLP is the primary evidence-to-answer transformer, architectures that strengthen
MLP computation (deeper MLPs, gated MLP pathways) may improve reasoning fidelity
more than attention modifications (more heads, longer context).

### 4.3 For Training

If $S_X$ measures internal reasoning state quality, it could serve as a training
objective: maximize internal evidence-grounded reasoning rather than surface-level
output correctness.

### 4.4 For the CoT Debate

Visible CoT may be an externalization of internal states, but the relationship is
partial. GPT-2 small shows zero ICI difference between faithful and unfaithful CoT
narratives — the model treats CoT as undifferentiated context. CoT increases $S_X$
by 25% but cannot repair misleading internal routing. This suggests visible CoT
should be understood as an amplifier of existing internal mechanisms, not as the
mechanism itself.

---

## 5. Comparison to Existing Frameworks

### Attention-as-Explanation
Finds that attention weights correlate with input importance. We go further: attention
routes evidence, but MLP computes answers. Routing ≠ computation.

### Chain-of-Thought Faithfulness
Studies whether visible CoT reflects internal processing. We study internal processing
directly and find that visible CoT may not always correspond.

### Mechanistic Interpretability (Circuits-style)
Identifies specific circuits for narrow tasks. We identify a general mechanism stage
structure (routing → transformation → storage → output) that applies across tasks.

### Probing
Trains classifiers on hidden states. We add permutation tests, random-state controls,
and token-count baselines to validate that probe accuracy reflects genuine reasoning
state, not artifacts.

---

## 6. Limitations and Open Questions

1. **Model scale**: Primary validation on GPT-2 family (82M–355M). Qwen2.5 (494M)
   shows consistent routing but needs full mechanistic replication.
2. **Task diversity**: 200 controlled reasoning samples. Real-world multi-hop QA
   (HotpotQA, FEVER, StrategyQA) needed for ecological validation.
3. **MLP mechanism detail**: Two-phase MLP pattern observed but not fully explained.
   What specific MLP features mediate the suppression-to-construction transition?
4. **Faithful/unfaithful CoT**: GPT-2 small shows no ICI discrimination. Larger,
   instruction-tuned models may differ — this is a model-scale question.
5. **Causal completeness**: Full causal mediation analysis (all layers, all components,
   all positions) remains for future work.

---

## 7. Contributions

1. **A mechanistic account of evidence-to-answer reasoning in Transformers**:
   QK routes evidence → MLP transforms → residual stores → logits answers.

2. **Evidence that MLP, not attention, is the dominant causal pathway** for
   evidence-to-answer transformation, with a two-phase (suppression → construction)
   layer-wise structure.

3. **Quantitative validation that residual streams encode reasoning states**
   ($S_X = 0.756$, $p < 0.0001$), with encoding capacity increasing with model depth.

4. **Discovery that internal reasoning mechanisms shift systematically with scale**:
   from routing-dominant (shallow) to state-encoding-dominant (deep).

5. **An open-source measurement pipeline** (Internal CoT Index) that maps all four
   mechanism stages to specific Transformer variables and provides causal validation
   through ablation, patching, and permutation testing.

---

## 8. Target Venues

- **Nature Machine Intelligence**: Strong fit — mechanistic AI discovery with
  cross-architecture validation and open-source pipeline.
- **Nature (main)**: Requires broader cross-disciplinary framing and larger-scale
  validation (see §6).
- **ICLR / NeurIPS**: Strong fit for mechanistic interpretability track.

---

*Document version: Foundation Validation Phase — May 2026.*
