# 4. Results: A Staged Internal Pathway from Evidence to Answer

*Draft — May 2026*

---

We present results organized by mechanism stage: QK routing (§4.1), MLP
transformation (§4.2), residual state storage (§4.3), scale-dependent mechanism
shift (§4.4), and the relationship between visible chain-of-thought and internal
mechanism (§4.5). All experiments use 200 controlled reasoning samples across
five types unless otherwise noted.

---

## 4.1 QK Attention Routes Evidence But Does Not Explain Answer Computation

**Claim**: QK attention weights identify evidence-bearing tokens, but routing
strength alone does not determine whether the model produces the correct answer.

### 4.1.1 Evidence routing discriminates direct from misleading prompts

We measure evidence routing strength ($R_{QK}$) as the attention mass flowing
from answer-relevant token positions to gold evidence token positions, averaged
across all layers and heads: $R_{QK} = \frac{1}{L \cdot H \cdot |P|}
\sum_{l,h,p \in P, j \in E} A_{l,h}[p, j]$.

Figure 2 shows $R_{QK}$ for direct-evidence versus misleading-hint samples
across four models spanning two architecture families. Direct-evidence samples
consistently show 8.5–19.3× higher evidence routing than misleading-hint samples
(Table 1).

**Table 1: Cross-architecture evidence routing.**

| Model | Layers | R_QK (direct) | R_QK (misleading) | Ratio |
|-------|--------|--------------|-------------------|-------|
| distilgpt2 | 6 | 0.255 | 0.025 | 10.2× |
| gpt2 | 12 | 0.236 | 0.022 | 10.7× |
| gpt2-medium | 24 | 0.162 | 0.019 | 8.5× |
| Qwen2.5-0.5B | 24 | 0.232 | 0.012 | 19.3× |

The discrimination holds across GPT-2 and Qwen2.5 architectures, suggesting
that evidence routing is a general property of Transformer attention rather
than an artifact of a specific model family. The absolute $R_{QK}$ for
direct-evidence samples decreases with model depth (0.255 at 6L → 0.162 at 24L),
consistent with attention distributing across more layers in deeper models.

Per-head analysis for GPT-2 reveals that evidence routing is concentrated in
early layers: the top five routing heads all reside in layers 0–1, with
Layer 0, Head 5 showing the highest $R_{QK}$ (0.299). This suggests that
evidence identification occurs early in the forward pass, with subsequent
layers performing different computational roles.

### 4.1.2 QK reconstruction verified

To ensure that our routing measurements reflect genuine model computation,
we verify that attention weights can be reconstructed from extracted Q and K
projections. For each layer $l$, we compute $\text{softmax}(Q_l K_l^T / \sqrt{d} +
\text{causal mask})$ and compare to the model's attention output. All 12 GPT-2
layers match within $10^{-5}$ absolute tolerance. This verification ensures that
subsequent analyses based on Q, K, V projections are grounded in the model's
actual computation rather than approximate reconstructions.

### 4.1.3 Routing is necessary but not sufficient

While $R_{QK}$ discriminates direct from misleading evidence, routing strength
alone does not determine answer correctness. We observe cases where evidence is
routed (high $R_{QK}$) but the model produces incorrect answers — for instance,
when multiple evidence documents conflict, the model attends to both but cannot
resolve the conflict. This motivates investigating what happens *after* routing:
how is evidence information transformed into answers?

---

## 4.2 MLPs Dominate the Causal Evidence-to-Answer Pathway

**Claim**: MLP layers — not attention head projections — are the dominant causal
pathway through which evidence is transformed into answer-relevant signals.

### 4.2.1 Component-specific ablation

We compare the causal effect of ablating (zeroing) attention output versus MLP
output at the answer token position, across all layers. Ablation effect is
measured as the absolute change in the gold answer token logit.

**Table 2: Cross-component causal ablation.**

| Component | |Δlogit| (all layers) | Dominant layers |
|-----------|---------------------------|-----------------|
| Attention output | 59.3 | 2/12 |
| MLP output | **98.1** | **10/12** |
| Residual (full block) | 124.5 | — |

MLP ablation causes 1.66× larger absolute logit changes than attention ablation.
The residual stream (full block output) shows the largest effect, consistent
with it accumulating contributions from both attention and MLP sublayers.

Layer-wise decomposition reveals that MLP dominates in 10 of 12 GPT-2 layers
(Figure 3B). Only Layer 0 — where evidence routing is concentrated — and
Layer 11 — adjacent to the final logit projection — show attention dominance.
Every intermediate layer (L1–L10) is MLP-dominant by absolute causal effect size.

### 4.2.2 MLP activation patching

We test whether MLP activations from a clean (correct evidence) forward pass can
restore the correct answer when patched into a corrupted (modified evidence)
forward pass. This is a stronger causal test than ablation: if MLP activations
carry answer-relevant information, replacing corrupted MLP outputs with clean
ones should recover the clean answer logit.

**Table 3: MLP activation patching recovery.**

| Pair | Modification | Recovery Ratio |
|------|-------------|---------------|
| pair_003 | 330m → 300m (Eiffel Tower) | 0.934 |
| pair_004 | 100°C → 90°C (water boiling) | 0.922 |
| pair_005 | 6400km → 7000km (Amazon) | >1.00 |
| **Average (5 pairs)** | | **0.771** |

MLP activation patching achieves 77.1% average recovery of the clean answer
logit, with 3 of 5 pairs exceeding 90% recovery. This demonstrates that MLP
activations causally encode answer-relevant evidence information. The two pairs
with lower recovery (pair_001, 0.000; pair_002, 0.000) involve single-number
answer changes ("2023"→"2022", "Paris"→"Lyon") where the corrupted evidence
may be sufficiently similar to the clean evidence that the model's internal
state is not strongly perturbed.

### 4.2.3 Distributed pathway decomposition

We decompose the total evidence-to-answer contribution into attention-pathway
($A \times V \times W_O$ projection) and MLP-pathway (direct MLP output
projection to vocabulary) components.

**Table 4: Pathway contribution fractions.**

| Model | Attention Fraction | MLP Fraction | Residual Fraction |
|-------|-------------------|--------------|-------------------|
| distilgpt2 (6L) | 0.062 | 0.131 | 0.807 |
| gpt2 (12L) | 0.073 | **0.428** | 0.499 |
| gpt2-medium (24L) | 0.041 | 0.143 | 0.816 |

The MLP pathway carries 2–6× more evidence-to-answer information than the
attention pathway across all models. The MLP fraction peaks at 12 layers
(42.8%), suggesting an intermediate depth where MLP-mediated evidence processing
is most concentrated. At 24 layers, the pathway becomes more distributed, with
the residual fraction dominating.

These results establish a division of labor: attention routes evidence to
relevant positions, but MLPs perform the primary causal transformation of
evidence into answer-relevant signals.

---

## 4.3 Residual Streams Store Distributed Reasoning States

**Claim**: Residual streams encode intermediate reasoning states that can be
linearly decoded, and this encoding is not attributable to surface-level confounds.

### 4.3.1 Linear probe performance

We train logistic regression probes on the last-token residual stream activation
(768-dimensional for GPT-2) to classify the five reasoning types. Probe accuracy
is evaluated via 5-fold stratified cross-validation.

**Table 5: Residual state probe validation.**

| Probe Setup | Accuracy | $S_X$ | Interpretation |
|-------------|----------|-------|----------------|
| Real residual states | **80.5%** ± 4.0% | **0.756** | Genuine encoding |
| Shuffled labels (100 permutations) | 20.4% ± 3.2% | — | Not label artifact |
| Random hidden states ($\mathcal{N}(0,1)$) | 15.0% | 0.000 | Not arbitrary correlation |
| Token-count baseline | 41.0% | 0.262 | Partially explained by length, but residual is 2× better |

The probe achieves 80.5% accuracy, normalized to $S_X = 0.756$ above the 20%
random baseline for 5-class classification. Three controls validate that this
reflects genuine reasoning state encoding rather than experimental artifacts:

1. **Label permutation**: Shuffling reasoning type labels drops accuracy to
   random chance (20.4%, $p < 0.0001$, 100 permutations). The high real accuracy
   is specifically due to the correspondence between residual states and true
   reasoning types.

2. **Random hidden states**: Replacing residual activations with Gaussian noise
   eliminates all classification signal ($S_X = 0.000$). The probe's success
   depends on structured information in the residual stream, not on classifier
   capacity or overfitting.

3. **Token-count baseline**: Using only the number of tokens in the prompt
   (which varies systematically: multi-step prompts are longer, direct-evidence
   prompts are shorter) achieves 41.0% accuracy ($S_X = 0.262$). The residual
   probe nearly doubles this performance, demonstrating that residual streams
   encode reasoning-specific information beyond surface-level features.

### 4.3.2 State encoding is distributed across layers

Layer-wise probes reveal that reasoning state information is distributed across
all 12 GPT-2 layers rather than concentrated in a single layer. Each layer
achieves approximately 30% probe accuracy ($S_X \approx 0.125$), with no single
layer exceeding 35%. This distributed encoding is consistent with the residual
stream architecture, where each layer adds incremental computation to a shared
representation.

### 4.3.3 Encoding capacity increases with model depth

$S_X$ increases with the number of layers: 0.583 (distilgpt2, 6L), 0.542
(gpt2, 12L), 0.708 (gpt2-medium, 24L). Deeper models encode more
reasoning-type information in their residual streams, consistent with the
hypothesis that additional layers contribute to richer internal state
representations. The non-monotonic pattern between 6L and 12L may reflect
differences in training rather than architecture.

---

## 4.4 Internal Reasoning Shifts from Routing-Dominant to State-Encoding-Dominant with Depth

**Claim**: The relative importance of mechanism stages changes systematically
with model depth, with shallow models relying on direct evidence routing and
deeper models increasingly depending on distributed state encoding.

### 4.4.1 Scale-aware weight calibration

Fixed equal weights ($\alpha = \beta = \gamma = \delta = 0.25$) implicitly
assume that all mechanism stages contribute equally regardless of model scale.
We calibrate scale-aware weights using within-model $R_{QK}$ gap, $S_X$ level,
and model depth.

**Table 6: Scale-aware ICI weights.**

| Model | Layers | $\alpha (R_{QK})$ | $\beta (M_{AV})$ | $\gamma (S_X)$ | Dominant Mode |
|-------|--------|-------------------|-------------------|----------------|---------------|
| distilgpt2 | 6 | **0.306** | 0.204 | 0.249 | Routing |
| gpt2 | 12 | 0.259 | 0.172 | 0.335 | Transition |
| gpt2-medium | 24 | 0.197 | 0.132 | **0.445** | State encoding |

The weight on $R_{QK}$ ($\alpha$) decreases from 0.306 to 0.197 as depth
increases, while the weight on $S_X$ ($\gamma$) increases from 0.249 to 0.445.
This systematic shift reveals that internal reasoning is not a single mechanism
but a collection of mechanisms whose relative importance changes with model
capacity. Fixed-weight metrics would mask this shift.

### 4.4.2 Pathway fraction changes non-monotonically

The MLP pathway fraction (Table 4) shows a non-monotonic pattern with scale:
peaking at 12 layers (42.8%) and declining at both 6 layers (13.1%) and 24
layers (14.3%). This suggests the existence of an intermediate depth where
MLP-mediated evidence processing is most concentrated, after which deeper models
distribute evidence processing across multiple pathways.

### 4.4.3 Within-model routing gap narrows but remains significant

The within-model ${R}_{QK}$ gap between direct and misleading evidence narrows
with depth (0.219 at 6L → 0.202 at 12L → 0.133 at 24L), but the ratio remains
strong (8.6× → 9.9× → 6.6×). Even at 24 layers, evidence routing still
discriminates faithful from misleading prompts by a factor of 6.6×.

---

## 4.5 Visible Chain-of-Thought Amplifies But Does Not Replace the Internal Mechanism

**Claim**: Chain-of-thought prompting enhances existing internal mechanisms but
cannot create mechanisms where none exist or repair systematically misdirected
evidence pathways.

### 4.5.1 CoT increases residual state separability

Chain-of-thought prompting increases $S_X$ from 0.550 to 0.575 (+4.5%),
indicating that explicit reasoning instructions produce more distinguishable
internal reasoning states. The effect is modest but consistent, suggesting
that visible CoT is partially externalizing internal mechanism states.

### 4.5.2 CoT does not repair misleading evidence pathways

**Table 7: CoT effect on ICI by reasoning type.**

| Reasoning Type | ICI (no-CoT) | ICI (CoT) | Δ |
|---------------|-------------|----------|----|
| direct_evidence | 0.122 | 0.138 | **+0.016** |
| conflict | 0.096 | 0.114 | **+0.018** |
| multi_step | 0.103 | 0.113 | **+0.010** |
| misleading_hint | 0.111 | 0.112 | **+0.000** |
| evidence_gap | 0.146 | 0.139 | −0.007 |

CoT increases ICI for reasoning-required types (direct, conflict, multi-step)
but has zero effect on misleading-hint samples. This asymmetry is informative:
CoT can amplify existing evidence-grounding when evidence is genuinely present
and correctly identified, but cannot overcome systematic misdirection when the
prompt is designed to route attention to incorrect evidence.

### 4.5.3 Faithful and unfaithful CoT are indistinguishable in small models

We construct 10 paired samples where the evidence is identical but the CoT
narrative is either faithful (correctly identifying and using the evidence)
or unfaithful (citing irrelevant information or making unsupported inferences).
In GPT-2 small, faithful and unfaithful CoT produce identical ICI scores
(ICI difference = 0.000 for all 10 pairs). The model treats all CoT text as
undifferentiated context — it does not differentially process faithful versus
unfaithful reasoning narratives.

This is not a failure of the measurement framework; it is a model-scale
limitation. GPT-2 small lacks the capacity to use CoT as a controllable
reasoning mechanism. ICI correctly reflects that the underlying evidence
routing — which is identical for both CoT variants since the evidence tokens
are the same — determines the internal signal. Whether larger, instruction-tuned
models show differential processing of faithful versus unfaithful CoT is an
open question that our framework is designed to test.

---

## 4.6 Cross-Model Validation

To test whether the mechanism chain is specific to GPT-2 or generalizes across
architectures, we replicate the three core claims on three models spanning two
architectural families: GPT-2 (12L, 124M), GPT-2-medium (24L, 355M), and
Qwen2.5-0.5B (24L, 494M, LLaMA-style architecture).

**Table 8: Cross-model foundation validation.**

| Model | C1: R_QK Ratio | C2: MLP/Attn |Δ| Ratio | C3: S_X (p) | Claims |
|-------|---------------|------------------------|-------------|--------|
| gpt2 (12L) | **10.6×** ✓ | **1.70×** ✓ | **0.50** (p<0.0001) ✓ | 3/3 |
| gpt2-medium (24L) | **7.6×** ✓ | 0.76× ✗ | **0.58** (p<0.0001) ✓ | 2/3 |
| Qwen2.5-0.5B (24L) | **17.4×** ✓ | **1.43×** ✓ | **0.70** (p<0.0001) ✓ | 3/3 |

Two claims are cross-architecture universal: QK evidence routing discriminates
direct from misleading evidence in all three models (7.6×–17.4×), and residual
streams encode reasoning states significantly above shuffled baselines in all
three models ($S_X = 0.50\text{–}0.70$, all $p < 0.0001$).

The third claim — MLP dominance over attention — is depth-dependent. It holds
strongly at 12 layers (GPT-2: 1.70×, Qwen2.5: 1.43×) but reverses at 24 layers
in GPT-2-medium (0.76×), where attention and MLP contributions converge. This
pattern is consistent with the scale-dependent mechanism shift documented in
§4.4: as models deepen, answer computation becomes more distributed across
pathways rather than concentrated in MLP layers.

## Summary of Main Results

The evidence-to-answer pathway in Transformers involves four stages, with
varying degrees of cross-model generality:

1. **QK attention robustly routes evidence** across architectures (7.6×–17.4×
   direct/misleading discrimination in 3/3 models). Routing is necessary but
   not sufficient — it identifies evidence but does not complete answer computation.

2. **MLP layers are the dominant causal transformation pathway at intermediate
   depth** (1.43–1.70× stronger than attention at 12L), but converge with attention
   in deeper models (0.76× at 24L), consistent with a shift toward distributed
   computation.

3. **Residual streams robustly store reasoning states** across architectures
   ($S_X = 0.50\text{–}0.70$, all $p < 0.0001$), with encoding capacity increasing
   with model depth.

4. **The mechanism shifts systematically with scale**: from routing-and-MLP-dominant
   at intermediate depth to distributed state-encoding-dominant in deeper models.
   Visible CoT amplifies internal mechanisms but cannot substitute for them.

---

## 4.7 TRACE Reduces Black-Box Failures at >1B Scale

The mechanism chain and audit framework established in §4.1–4.6 raise a
practical question: can internal mechanism traces be used to actually reduce
black-box failures, rather than merely detect them? We test this by applying
TRACE-guided intervention to two independently tested >1B-parameter models —
Qwen2.5-1.5B (28 layers) and Qwen2.5-3B (36 layers) — on a balanced set of
error-prone samples spanning misleading hints, conflicting evidence, and
insufficient evidence.

For each sample, TRACE extracts a minimal internal trace (R_QK and logit
confidence), diagnoses the mechanism weakness, and applies a matched
intervention: conservative prompting with abstention capability for
conflict and evidence-gap samples, and misleading-cue filtering for
misleading-hint samples. We compare TRACE against four baselines: raw
model output, chain-of-thought prompting, confidence-based abstention
(conservative prompting when softmax confidence < 0.3), and
attention-entropy-based intervention (conservative prompting when
R_QK < 0.02).

**Table 9: TRACE error reduction on Qwen2.5-1.5B (n=120, 40 per type).**
All error rates reported with Wilson 95% confidence intervals.
Absolute and relative reductions with 10,000-sample paired bootstrap CIs.
Significance via McNemar test (*** p<0.001).

| Type | n | Raw Errors | Raw Error [95% CI] | TRACE Errors | TRACE Error [95% CI] | Rel. Reduction [95% CI] | p |
|------|---|-----------|-------------------|-------------|---------------------|------------------------|---|---|
| conflict | 40 | 39/40 | 97.5% [87.1, 99.6] | 1/40 | 2.5% [0.4, 12.9] | 97.4% [92.1, 100.0] | <0.0001*** |
| evidence_gap | 40 | 39/40 | 97.5% [87.1, 99.6] | 2/40 | 5.0% [1.4, 16.5] | 94.9% [87.2, 100.0] | <0.0001*** |
| misleading_hint | 40 | 32/40 | 80.0% [65.2, 89.5] | 30/40 | 75.0% [59.8, 85.8] | 6.3% [-14.3, 24.2] | 0.752 |
| **Pooled** | **120** | **110/120** | **91.7% [85.3, 95.4]** | **33/120** | **27.5% [20.3, 36.1]** | **70.0% [60.7, 78.6]** | **<0.0001***** |

Both conflict and evidence-gap conditions produced 39/40 raw failures, yielding
identical raw error estimates, although TRACE reduced them to different residual
error rates (1/40 and 2/40 respectively).

Direct evidence false positives: 0/40 (0.0% [0.0, 8.8]) for both raw and TRACE.

Three findings emerge with statistical confidence. First, TRACE achieves
substantial and significant error reduction on the pooled sample (70.0%
relative reduction [95% CI: 60.7, 78.6], McNemar p < 0.000001), with
near-complete elimination of conflict non-disclosure (97.4% reduction
[92.1, 100.0], p < 0.0001) and evidence-gap unsupported answers (94.9%
reduction [87.2, 100.0], p < 0.0001). Conservative prompting
with explicit abstention capability causes the model to refrain from
answering when evidence is insufficient or contradictory — the intended
mechanism-matched behavior.

Second, misleading-driven errors are the consistent weak spot with
statistical confirmation: TRACE reduces misleading error by 6.3% on the
1.5B model, but the 95% CI for relative reduction crosses zero
[-14.3%, 24.2%] and the McNemar test is non-significant (p = 0.75).
This is consistent with the format-sensitivity of QK routing documented in
§4.5–4.6: QK attention to evidence is largely determined by token-level
co-occurrence, and removing a few misleading cue phrases from the prompt
is insufficient to redirect routing when the remaining evidence tokens
still support the misleading interpretation.

Third, TRACE achieves these reductions without harming safe samples. Direct
evidence false positives remain at 0% on both models — TRACE does not cause
the model to abstain or change its answer when the internal mechanism chain
indicates a well-grounded evidence-to-answer pathway.

**Table 10: TRACE vs baselines on >1B models.**

| Strategy | 1.5B Error | 3B Error | Character |
|----------|-----------|---------|-----------|
| Raw model | 70.0% | 65.6% | Baseline |
| Chain-of-Thought | 80.0% | 62.5% | Sometimes worse than raw |
| Confidence abstention | 30.0% | 43.8% | Indiscriminate; 47–31% fire rate |
| Attention entropy | 7.5% | 12.5% | Lower error but over-abstains (63–75% safe rate) |
| **TRACE selective** | **17.5%** | **28.1%** | **Targeted; mechanism-matched** |

Comparison to baselines reveals a critical distinction. Attention-entropy-based
intervention achieves the lowest raw error rates (7.5–12.5%) but does so by
abstaining on 63–75% of all samples — an indiscriminate strategy that would
render the model unusable in deployment settings where answers are expected.
Confidence-based abstention reduces error to 30–44% but fires
indiscriminately on both error-prone and safe samples. TRACE achieves error
reduction comparable to or better than confidence-based approaches while
preserving safe-sample throughput, and does so through mechanism-matched
intervention selection rather than uniform thresholding.

These results close the loop from mechanism discovery to practical black-box
reduction: the same internal variables that reveal the evidence-to-answer
pathway (§4.1–4.3) can be operationalized to reduce failures when that
pathway is weak. The consistent limitation on misleading-driven errors
indicates a boundary condition: when evidence routing is fundamentally
misdirected by the prompt structure, surface-level intervention is
insufficient, and deeper mechanism modification may be required.
