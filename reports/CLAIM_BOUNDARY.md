# Claim Boundary Table — Final (Nature Manuscript)

*May 2026. Updated with TRACE-Scale results.*

## CAN CLAIM (High Confidence)

| # | Claim | Evidence |
|---|-------|----------|
| C1 | Evidence-to-answer follows a staged mechanism (QK→MLP→X_l→logits) | Cross-model (3 models), cross-format (5 formats), causal (ablation + patching) |
| C2 | QK attention routes evidence, with 7.6–19.3× direct/misleading discrimination | 3/3 models, cross-architecture |
| C3 | MLP is the dominant causal pathway (1.4–3.0× over attention) | 54/54 samples, 5 formats |
| C4 | Residual streams encode reasoning states (S_X=0.50–0.99, all p<0.0001) | Permutation, random, token-count controls |
| C5 | Internal mechanism shifts from routing-dominant to state-encoding-dominant with depth | 3 depth points, GPT-2 family |
| C6 | TRACE enables selective, mechanism-matched intervention | 5 error types, targeted strategies |
| C7 | TRACE reduces black-box failures by 57–75% on two >1B models | Qwen2.5-1.5B + Qwen2.5-3B |
| C8 | TRACE eliminates conflict non-disclosure and evidence-gap unsupported answers at >1B | -100% on both models |
| C9 | TRACE achieves reductions without increasing false positives | 0% false positive on both models |

## CAN DISCUSS (Medium Confidence)

| # | Claim | Qualification |
|---|-------|---------------|
| D1 | Mechanism generalizes across all Transformer architectures | Tested on GPT-2 + Qwen2.5; LLaMA/Mistral pending |
| D2 | TRACE outperforms all baselines on all error types | Outperforms confidence; attention entropy over-abstains; misleading remains weak |
| D3 | MLP two-phase computation (early/late suppression, middle construction) | Observed in GPT-2 12L; cross-model validation pending |
| D4 | Error reduction scales with model size | 2 data points; larger-scale trend unconfirmed |
| D5 | Visible CoT is an amplifier, not the mechanism | Tested on GPT-2 family; instruction-tuned models may differ |

## MUST NOT CLAIM

| # | Overclaim | Why |
|---|-----------|-----|
| X1 | Full reverse-engineering of Transformer reasoning | Only structural skeleton identified |
| X2 | TRACE eliminates all black-box failures | Misleading remains partially resistant |
| X3 | TRACE works universally across all models and tasks | 5 architectures, 5 formats, controlled tasks |
| X4 | TRACE is optimal | Heuristic weights, threshold-based diagnosis |
| X5 | Two-phase MLP is universal | Low confidence, single-model observation |
| X6 | TRACE replaces need for external evaluation | Complementary tool, not replacement |

---

## CAN CLAIM (High Confidence)

These are supported by direct experimental evidence across multiple models or controls.

| # | Claim | Evidence | Cross-check |
|---|-------|----------|-------------|
| C1 | QK attention routes evidence to relevant token positions | R_QK(direct) / R_QK(misleading) = 8.5–19.3× | GPT-2 + Qwen2.5 |
| C2 | QK attention weights are verifiably correct | softmax(QK^T/√d) ≈ A_model, all 12 layers, tol 10^{-5} | Direct reconstruction |
| C3 | MLP output ablation causes larger logit changes than attention output ablation | |Δlogit|: MLP=98.1 vs Attn=59.3 (1.66×) | Consistent across 5 samples |
| C4 | MLP activation patching can recover correct answer from corrupted evidence | 77.1% avg recovery, >90% for 3/5 pairs | 15 corrupted pairs |
| C5 | Residual streams encode information about reasoning type | S_X = 0.756, p < 0.0001 | Permutation, random, token-count controls |
| C6 | S_X is not a template artifact | Shuffled labels = 20.4%, token-count = 41.0% < 80.5% | 3 negative controls |
| C7 | ICI ranks direct_evidence above misleading_hint | Consistent across all models and versions (v0.1–v0.6) | 200 samples, 5 models |
| C8 | CoT prompts increase S_X | ΔS_X = +0.025 (no-CoT vs CoT) | GPT-2 family |
| C9 | CoT prompts do not improve ICI for misleading-hint samples | ΔICI = 0.000 for misleading | Consistent across samples |

---

## CAN DISCUSS (Medium Confidence, Needs Qualification)

These are supported by evidence but require explicit qualification about scope or confidence.

| # | Claim | Evidence | Qualification |
|---|-------|----------|---------------|
| D1 | Evidence-to-answer processing follows a staged mechanism (QK→MLP→X_l→logits) | Multiple component measurements align | "We identify a structural skeleton, not a complete mechanism" |
| D2 | Internal mechanism shifts from routing-dominant to state-encoding-dominant with depth | α↓, γ↑ with 3 depth points | "Observed in GPT-2 family (6L–24L); generality to other architectures and larger scales remains to be tested" |
| D3 | MLP pathway carries more evidence-to-answer information than attention pathway | MLP fraction 13–43% vs attention 4–7% | "Measured via logit projection; relative contribution may vary with task and model" |
| D4 | GPT-2 small does not differentially process faithful vs unfaithful CoT | Identical ICI for 10 pairs | "May be a model-scale limitation; larger instruction-tuned models may differ" |
| D5 | MLP layers show a two-phase computation pattern | Layer-wise ablation: middle construct, late suppress | "Observed in GPT-2 12L; cross-model validation pending" |
| D6 | Visible CoT is an amplifier of internal mechanisms, not the mechanism itself | CoT boosts S_X but can't fix misleading | "Tested on GPT-2 family; relationship may differ in models with stronger instruction following" |

---

## MUST NOT CLAIM (Insufficient Evidence)

These would be overclaims given current evidence. We must explicitly avoid them.

| # | Overclaim | Why We Cannot Claim It |
|---|-----------|----------------------|
| X1 | "We have fully reverse-engineered Transformer reasoning" | We identify a structural skeleton; detailed computation within each stage (especially MLP feature-level mechanisms) is not mapped |
| X2 | "The mechanism chain is universal across all Transformers" | Tested on GPT-2 family (82M–355M) and Qwen2.5-0.5B; larger models, LLaMA, Mistral, Gemma not yet validated |
| X3 | "ICI is a general-purpose reasoning quality metric" | ICI is validated on 200 controlled reasoning samples; real-world QA generalization not tested |
| X4 | "MLP always dominates attention in all models" | MLP dominance strongest at 12L; at 24L, attention and MLP contributions converge |
| X5 | "Two-phase MLP is a universal Transformer property" | Only observed clearly in single-sample per-layer GPT-2 analysis; averaged results show weak signal |
| X6 | "CoT is not useful for reasoning" | We show CoT amplifies internal mechanisms; it does help, just cannot repair fundamentally misleading evidence paths |
| X7 | "Our method replaces the need for chain-of-thought" | ICI measures internal state; it does not improve model output or replace CoT as a prompting strategy |
| X8 | "All attention heads are evidence routers" | Only a subset of heads show high R_QK; many heads serve other functions |

---

## Explicit Scope Statement (Include in Paper)

The following statement should appear in the Discussion:

> We do not claim to have fully reverse-engineered Transformer reasoning. Our
> contribution is the identification of a structural skeleton for evidence-to-answer
> computation — QK routing → MLP transformation → residual state storage → logit
> projection — and the demonstration that this skeleton can be measured,
> decomposed, and causally validated. The detailed feature-level computation
> within each stage, the universality of the two-phase MLP pattern, and the
> behavior of this mechanism at scales beyond 500M parameters remain open
> questions for future work. What we establish is that Transformer reasoning is
> not attention-only, that MLPs play a larger causal role in evidence-to-answer
> transformation than previously emphasized, and that internal reasoning traces
> can be quantified without relying on external chain-of-thought text.

---

## Reviewer Anticipation

### Likely Criticism 1: "This is only shown on small models"

**Response**: Our goal is mechanism identification, not scale demonstration. GPT-2-size
models allow complete internal variable access (Q, K, V, MLP, residual) that is
prohibitively expensive at 7B+ scale. The cross-architecture replication (GPT-2 +
Qwen2.5) provides initial evidence of generality. We explicitly flag scale as a
limitation and future work direction.

### Likely Criticism 2: "The mechanism might be task-specific"

**Response**: We test five distinct reasoning types (direct, conflict, multi-step,
evidence gap, misleading) and find consistent patterns. However, we acknowledge
that all tasks are controlled synthetic reasoning tasks. Real-world QA generalization
is future work.

### Likely Criticism 3: "Probes can overfit; S_X might not reflect genuine reasoning state"

**Response**: We include three controls: (1) label permutation test (p < 0.0001),
(2) random hidden states (S_X = 0.000), (3) token-count baseline (S_X = 0.262 < 0.756).
These rule out the most common probe confounds. We acknowledge that probes measure
linear separability, not necessarily causal necessity.

### Likely Criticism 4: "ICI weights are arbitrary"

**Response**: v0.5 addresses this with scale-aware calibration. We show that
fixed 0.25 weights mask scale-dependent mechanism shifts. We recommend reporting
per-component scores alongside any aggregate ICI, and we provide calibrated weights
as a default, not a claim of optimality.

---

## One-Sentence Version for Abstract

> We identify a staged internal mechanism for evidence-to-answer computation in
> Transformers — QK routes evidence, MLPs transform it, residual streams store
> intermediate states — and introduce a measurement framework (Internal CoT Index)
> to quantify this pathway, finding that visible chain-of-thought is a partial
> externalization rather than the mechanism itself.
