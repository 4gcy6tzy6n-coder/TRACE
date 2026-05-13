# Figure Plan — Final (Nature Manuscript)

*May 2026. Updated with TRACE-Scale results.*

---

## Figure 1: Mechanism Overview

**Type**: Schematic / diagram

**Content**:
```
┌─────────────────────────────────────────────────────┐
│              EVIDENCE-TO-ANSWER PATHWAY              │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │    QK    │───▶│   MLP    │───▶│ Residual │      │
│  │ Routes   │    │Transforms│    │  Stores  │      │
│  │ Evidence │    │ Evidence │    │  State   │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │               │               │             │
│       ▼               ▼               ▼             │
│  R_QK = 0.23     M_AV = 0.85     S_X = 0.76        │
│  (direct)        (direct)        (p < 0.0001)      │
│  R_QK = 0.02     M_AV = 0.06     C_do = 0.60       │
│  (misleading)    (misleading)    (ablation)         │
│                                                     │
│                    ┌──────────┐                     │
│                    │  Logits  │                     │
│                    │ Produce  │                     │
│                    │  Answer  │                     │
│                    └──────────┘                     │
└─────────────────────────────────────────────────────┘
```

**Caption**: Overview of the evidence-to-answer mechanism chain in Transformers.
QK attention routes evidence to relevant token positions (measured by R_QK),
MLP layers causally transform evidence into reasoning states (measured by M_AV and
distributed pathway decomposition), residual streams store intermediate representations
(measured by S_X probe), and logits produce the final answer. Direct-evidence samples
show consistently stronger internal signals than misleading-hint samples across all stages.

**Data source**: v0.2 (QK verification), v0.3 (S_X), v0.5 (M_AV), v0.6 (MLP causal)

---

## Figure 2: R_QK — Evidence Routing Across Architectures

**Type**: Grouped bar chart

**Content**: R_QK for direct_evidence vs misleading_hint across 4 models.
X-axis: model (distilgpt2, gpt2, gpt2-medium, Qwen2.5-0.5B)
Y-axis: R_QK score
Two bars per model: direct_evidence (blue), misleading_hint (red)
Ratio annotation above each pair: "10.2×", "10.7×", "8.5×", "19.3×"

**Caption**: QK evidence routing strength (R_QK) for direct-evidence versus
misleading-hint samples across GPT-2 and Qwen2.5 architectures. R_QK measures
attention mass from answer to evidence token positions. The 8.5–19.3× ratio
demonstrates that QK attention reliably discriminates evidence-grounded from
misleading prompts, and this discrimination generalizes across model families.

**Subfigure B** (optional): Per-head R_QK heatmap (12×12) for GPT-2, showing
evidence routing concentrated in early layers (Layer 0, heads 1,3,4,5).

**Data source**: v0.4 (cross-architecture), v0.2 (per-head)

---

## Figure 3: MLP Dominance — Causal Ablation Comparison

**Type**: Dual panel

**Panel A**: Bar chart — |Δlogit| for attention ablation vs MLP ablation
(MLP: 98.1, Attention: 59.3, ratio 1.66×)

**Panel B**: Layer-wise line chart — MLP Δlogit (solid) vs attention Δlogit (dashed)
across 12 GPT-2 layers. Shaded region: MLP-dominant layers (10/12).
Annotation: "Middle layers (L3-7): construction phase", "Late layers (L8-11): suppression phase"

**Caption**: Causal comparison of attention-output versus MLP-output ablation
at the answer token position. (A) MLP ablation causes 1.66× larger absolute logit
changes than attention ablation. (B) Layer-wise analysis reveals MLP dominance in
10 of 12 layers, with only layer 0 (initial routing) and layer 11 (final output)
showing attention dominance. Middle layers (3-7) construct the answer signal
(positive Δ after ablation), while late layers (8-11) suppress premature output
(negative Δ after ablation).

**Data source**: v0.6 (MLP causal)

---

## Figure 4: S_X — Residual Reasoning State Probe

**Type**: Multi-panel validation figure

**Panel A**: Bar chart — probe accuracy comparison:
Real residual (80.5%), Shuffled labels (20.4%), Random states (15.0%),
Token-count baseline (41.0%). Dashed line: random baseline (20%).

**Panel B**: Confusion matrix — 5×5 for reasoning type classification.

**Panel C**: Layer-wise S_X line chart — S_X per layer (all ~0.125 for GPT-2),
showing distributed encoding across depth.

**Panel D**: S_X vs model depth scatter — 3 points (6L, 12L, 24L) with upward trend.

**Caption**: Residual stream encoding of reasoning state. (A) Linear probe accuracy
with controls: real residual states achieve 80.5% (S_X = 0.756), while shuffled
labels drop to random (20.4%, p < 0.0001), random hidden states produce S_X = 0.000,
and token-count baseline reaches only 41.0%. (B) The probe distinguishes all five
reasoning types. (C) Reasoning state information is distributed across all layers.
(D) Encoding capacity increases with model depth.

**Data source**: v0.3 (permutation), v0.4 (layer-wise), v0.5 (scale trend)

---

## Figure 5: Mechanism Shifts with Scale

**Type**: Stacked area chart or ternary plot

**Panel A**: Weight evolution — α(R_QK), β(M_AV), γ(S_X) as stacked bars
for 6L, 12L, 24L models. α shrinks, γ grows.

**Panel B**: Pathway fraction — attention fraction, MLP fraction, residual fraction
as stacked bars for same models. MLP fraction peaks at 12L (42.8%).

**Caption**: Scale-dependent mechanism shift. (A) Scale-aware ICI weights show
systematic transition from R_QK-dominant (6L: α=0.306) to S_X-dominant
(24L: γ=0.445) as model depth increases. (B) Distributed pathway decomposition
reveals MLP-mediated evidence processing peaks at intermediate scale (12L: 42.8%),
suggesting an optimal depth for MLP-focused computation before distributed processing
takes over in deeper models.

**Data source**: v0.5 (scale-aware calibration, distributed M_AV)

---

## Figure 6: Visible CoT vs Internal Mechanism

**Type**: Dual panel

**Panel A**: Bar chart — ICI for direct_evidence and misleading_hint under
no-CoT and CoT conditions. CoT increases ICI for direct (+0.016) but not
misleading (+0.000).

**Panel B**: Scatter plot — ICI(faithful CoT) vs ICI(unfaithful CoT) for
10 pairs, with identity line. All points lie on the diagonal (GPT-2 small
cannot discriminate).

**Caption**: Relationship between visible chain-of-thought and internal mechanism.
(A) CoT prompting increases ICI for direct-evidence samples but has zero effect on
misleading-hint samples — CoT amplifies existing internal mechanisms but cannot
repair misleading evidence pathways. (B) GPT-2 small produces identical ICI scores
for faithful and unfaithful CoT narratives, indicating that visible CoT text is not
differentially processed as a reasoning mechanism at this scale.

**Data source**: v0.3 (CoT comparison), v0.3 (faithful/unfaithful)

---

## Figure 7: TRACE Error Reduction at >1B (NEW)

**Type**: Dual bar chart + per-type breakdown

**Panel A**: Grouped bar — Error rate for Raw vs TRACE on Qwen2.5-1.5B and Qwen2.5-3B.
Two model groups, two bars each (Raw, TRACE). Values: 70→17.5%, 65.6→28.1%.

**Panel B**: Per-type breakdown — Conflict, Evidence Gap, Misleading error rates.
Before/after arrows for each type on each model.

**Caption**: TRACE-guided intervention reduces black-box reasoning failures on
two independently tested >1B-parameter models. Total error rates decrease by
75.0% (1.5B) and 57.2% (3B). Conflict non-disclosure and evidence-gap
unsupported answers are nearly eliminated (100% and 71–100% reduction).
Misleading-driven errors remain partially resistant, consistent with the
format-sensitivity of QK-based evidence routing. Direct-evidence false positives
remain at 0% across both models.

## Supplementary Figures

### S1: Q/K/V Reconstruction Verification
12-panel grid showing softmax(QK^T/√d) vs model attention for each layer.
All 12 pass within 10^{-5} tolerance.

### S2: Activation Patching Controls
Bar chart showing recovery ratio for 6 control types:
correct_residual (HIGH), random_clean (LOW), unrelated_type (LOW),
same_type_wrong (LOW), evidence_positions (MED), answer_position (MED).

---

## Table Plan

### Table 1: Dataset Composition
5 reasoning types, 200 samples, per-type examples.

### Table 2: Cross-Architecture R_QK
4 models, direct vs misleading, ratio, gap.

### Table 3: Causal Component Comparison
MLP vs attention |Δlogit|, MLP-dominant layers, patch recovery.

### Table 4: S_X Validation Matrix
Real, shuffled, random, token-count — accuracy and S_X.

### Table 5: Scale-Aware Weights
α, β, γ, δ per model depth.

### Table 6: CoT Effects
ICI and S_X under no-CoT, CoT, faithful CoT, unfaithful CoT.
