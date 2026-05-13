# 3. Methods: Measuring the Evidence-to-Answer Pathway

*Draft — May 2026*

---

The methods are organized around the proposed mechanism chain: QK routing,
MLP transformation, residual-state storage, and logit-level answer production.
Each measurement is designed to test one stage of this chain. We describe the
experimental setting (§3.1), internal variable extraction (§3.2), the four-stage
measurement framework (§3.3), causal intervention protocols (§3.4), probe
validation controls (§3.5), and scale-aware calibration (§3.6).

---

## 3.1 Experimental Setting and Controlled Reasoning Tasks

### 3.1.1 Models

We evaluate on GPT-2 family models (distilgpt2: 6 layers, 82M parameters;
gpt2: 12 layers, 124M; gpt2-medium: 24 layers, 355M) and Qwen2.5-0.5B
(24 layers, 494M). GPT-2 models use fused QKV projections (Conv1D `c_attn`);
Qwen2.5 uses separate `q_proj`, `k_proj`, `v_proj` linear layers. Pythia models
(70M, 160M, 410M) were also tested but exhibited float16 NaN overflow in deep
layers, documented as a precision limitation (§3.2). All models are loaded from
standard HuggingFace checkpoints and evaluated in inference mode.

### 3.1.2 Controlled reasoning dataset

We construct 200 samples across five reasoning types (40 per type). These tasks
are not designed to benchmark model accuracy; they provide controlled settings
in which the relationship between evidence and answer is known, enabling
precise measurement of the internal evidence-to-answer pathway.

| Type | Description | Example |
|------|-------------|---------|
| `direct_evidence` | Answer directly stated in one document | "The model was released in 2023." → Q: "When?" → A: "2023" |
| `evidence_gap` | Answer cannot be determined from given evidence | Location + topic given, Q asks for attendees → A: "Cannot determine" |
| `multi_step` | Requires arithmetic or multi-step inference | Width 10cm, length 15cm → Q: "Area?" → A: "150 sq cm" |
| `conflict` | Multiple documents give contradictory information | Doc A: $5B revenue, Doc B: $3B → A: "Cannot determine" |
| `misleading_hint` | Distractor claims alongside hard evidence | Ads claim effectiveness, clinical trial shows no effect |

Each sample contains: evidence documents, a question, a gold answer, a gold
evidence span (the specific text substring that supports the answer), annotated
reasoning steps, and a reasoning type label. The gold evidence span enables
precise token-level mapping from evidence text to tokenizer indices (§3.2.3).

---

## 3.2 Extracting Transformer Internal Variables

### 3.2.1 Architecture-aware extraction

We extract per-layer, per-head Q, K, V projections by registering forward hooks
on the model's attention projections. For GPT-2, which uses a fused `c_attn`
projection outputting $[Q; K; V] \in \mathbb{R}^{batch \times seq \times 3d}$,
we split the concatenated output and reshape to $[batch, heads, seq, d_{head}]$.
For Qwen2.5, which uses separate `q_proj`, `k_proj`, `v_proj`, we hook each
projection independently and reshape. For Pythia (GPT-NeoX), we hook the fused
`query_key_value` projection.

Architecture detection is automatic: we read `model.config.architectures` to
determine the model family and register appropriate hooks. The extraction
pipeline also captures attention weights (post-softmax), hidden states (post-block
residual stream), MLP activations (post-MLP sublayer), and logits.

### 3.2.2 QK reconstruction verification

To ensure extracted Q and K correspond to actual model computation, we verify
that attention weights can be reconstructed. For each layer $l$:

$$A'_l = \text{softmax}\left(\frac{Q_l K_l^T}{\sqrt{d_{head}}} + \text{causal mask}\right)$$

We compare $A'_l$ to the model's output attention weights $A_l$ using
`torch.allclose(atol=10^{-5})`. All 12 GPT-2 layers pass. For the Attention
Mask Hook (§3.4.1), we additionally verify that an identity pass (empty mask)
produces logits matching the original forward pass within 0.01.

### 3.2.3 Token span mapping

Evidence spans and answer spans are mapped to token indices using character-offset
alignment. For each sample, we locate the gold evidence span within the full prompt
text via case-insensitive substring matching, then map character positions to token
indices using the tokenizer's `return_offsets_mapping`. Answer positions are found
via sliding-window token matching. When answer text cannot be located in the prompt
(e.g., in generation settings), the last token position is used as the answer
position.

---

## 3.3 Internal CoT Measurement Framework

We decompose the evidence-to-answer pathway into four measurable stages.
Each stage corresponds to a specific internal variable or intervention,
enabling quantitative comparison across samples, models, and conditions.

### 3.3.1 $R_{QK}$: Evidence routing

$R_{QK}$ measures the attention mass flowing from answer-relevant positions
to evidence token positions, quantifying the QK routing stage:

$$R_{QK} = \frac{1}{L \cdot H \cdot |P|} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{p \in P} \sum_{j \in E} A_{l,h}[p, j]$$

where $L$ is the number of layers, $H$ the number of heads, $P$ the set of
answer-relevant token positions, $E$ the set of gold evidence token positions,
and $A_{l,h}[p, j]$ the attention weight from query position $p$ to key
position $j$ in layer $l$, head $h$.

$R_{QK} \in [0, 1]$, with higher values indicating stronger evidence routing.
We also compute per-layer and per-head $R_{QK}$ for fine-grained analysis.

### 3.3.2 $M_{AV}$: Evidence message contribution

$M_{AV}$ measures the contribution of evidence value vectors, routed through
attention, to the answer token logit. This quantifies the MLP transformation
stage — specifically, whether evidence information survives the attention-to-output
projection and reaches the vocabulary:

$$M_{AV} = \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{j \in E} A_{l,h}[p, j] \cdot V_{l,h}[j] \cdot W_O^{(h)} \cdot W_U[y]$$

where $V_{l,h}[j]$ is the value vector for evidence token $j$, $W_O^{(h)}$ is
the output projection slice for head $h$, and $W_U[y]$ is the unembedding vector
for the gold answer token $y$. The result is normalized to $[0, 1]$ using
sigmoid scaling.

For models where QKV extraction is unavailable or unreliable, a proxy $M_{AV}$
can be computed via evidence removal logit drop: $\Delta\text{logit} =
\text{logit}_y(\text{full prompt}) - \text{logit}_y(\text{prompt without evidence})$.
We report true $M_{AV}$ where possible and note when the proxy is used.

**Distributed $M_{AV}$**: We further decompose the total evidence-to-answer
contribution into attention-pathway ($A \times V \times W_O$ only) and
MLP-pathway (direct MLP output projection to vocabulary) components,
enabling comparison of which pathway carries more evidence information.

### 3.3.3 $S_X$: Residual-state separability

$S_X$ measures whether residual stream activations encode information about
the reasoning type, quantifying the residual state storage stage:

$$S_X = \frac{\text{probe\_accuracy} - 1/K}{1 - 1/K}$$

where $K = 5$ is the number of reasoning types. A logistic regression probe
is trained on last-token residual stream activations (768-dimensional for GPT-2)
to classify the five reasoning types. Accuracy is evaluated via 5-fold stratified
cross-validation. $S_X$ is normalized above the random baseline: $S_X = 0$
indicates chance-level encoding, $S_X = 1$ indicates perfect separability.

We also train per-layer probes to localize where in the network reasoning
state information is encoded, and per-layer MLP probes for comparison with
residual stream probes (§3.5).

### 3.3.4 $C_{do}$: Causal intervention sensitivity

$C_{do}$ measures how strongly intervening on internal pathways changes the
answer logit, quantifying the causal dependence of the answer on the identified
mechanism:

$$C_{do} = \frac{1}{2}\left(f(\Delta_{\text{token}}) + f(\Delta_{\text{attention}})\right)$$

where $f(x) = 2/(1 + e^{-x/5}) - 1$ normalizes logit drops to $[0, 1]$.
$\Delta_{\text{token}}$ is the logit drop from removing the evidence span
from the input text. $\Delta_{\text{attention}}$ is the logit drop from
masking attention weights from answer positions to evidence positions across
all layers and heads (§3.4.1).

### 3.3.5 ICI aggregation

The Internal CoT Index aggregates the four stage measurements:

$$ICI = \alpha R_{QK} + \beta M_{AV} + \gamma S_X + \delta C_{do}$$

Default weights are equal ($\alpha = \beta = \gamma = \delta = 0.25$).
Scale-aware calibration (§3.6) adjusts weights based on model depth and
within-model measurement characteristics.

---

## 3.4 Causal Intervention and Patching

### 3.4.1 Component-specific ablation

We perform targeted ablation of specific Transformer components by registering
forward hooks that zero the component's output at a specified token position:

- **Attention output ablation**: Zero the attention sublayer output
  (`attn_output` at specified position) before it enters the residual stream.
- **MLP output ablation**: Zero the MLP sublayer output at specified position.
- **Residual ablation**: Zero the full block output at specified position.

For all ablations, the original forward pass logit is compared to the ablated
logit: $|\Delta\text{logit}| = |\text{logit}_{\text{orig}} - \text{logit}_{\text{abl}}|$.
All-layer ablation zeros the component across all layers simultaneously.
Layer-wise ablation zeros one layer at a time.

**True attention masking**: For $C_{do}$ attention ablation, we use an
`AttentionMaskHook` that overrides the attention computation in each layer.
The hook intercepts hidden states, recomputes QKV from the layer's own weights,
and sets attention scores from answer positions to evidence positions to
$-\infty$ before softmax. This provides genuine attention-level masking,
distinct from the embedding-zeroing approach used in prior work. Identity
verification (empty mask → identical logits) confirms correctness.

### 3.4.2 Activation patching

Activation patching tests whether internal activations from a clean (correct
evidence) forward pass can restore the correct answer when inserted into a
corrupted (modified evidence) forward pass:

1. Run clean forward pass, cache hidden states (or MLP outputs) at each layer.
2. Run corrupted forward pass.
3. Run corrupted forward pass with `ResidualPatchHook`, which replaces block
   outputs (or sub-component outputs) at specified layers/positions with the
   cached clean activations.
4. Measure recovery ratio: $(\text{logit}_{\text{patched}} - \text{logit}_{\text{corrupted}})
   / (\text{logit}_{\text{clean}} - \text{logit}_{\text{corrupted}})$.

Recovery ratio of 1.0 indicates full restoration of the clean answer; 0.0
indicates no effect.

### 3.4.3 Patching controls

We implement six control conditions to rule out confounds:

| Control | Patch Source | Expected Recovery | Rules Out |
|---------|-------------|-------------------|-----------|
| Correct residual | Clean version of same sample | High | — |
| Random clean | Random different clean sample | Low | Any perturbation causes recovery |
| Unrelated type | Different reasoning type | Low | Distribution shift from patching |
| Same-type wrong | Same type, different answer | Low-Med | Type-level features, not answer-specific |
| Evidence positions only | Same clean, only evidence tokens | Med-High | Non-evidence positions carry signal |
| Answer position only | Same clean, only answer token | Med | Answer-position-specific effects |

### 3.4.4 Corrupted pairs

For patching experiments, we construct clean/corrupted sample pairs by modifying
the evidence to change the gold answer. For direct-evidence samples, this means
changing the key value (e.g., "released in 2023" → "released in 2022"). For
misleading-hint samples, this means flipping the hard evidence to support the
misleading claim. Fifteen pairs were constructed across reasoning types.

---

## 3.5 Probe Validation and Confound Controls

### 3.5.1 Cross-validation

All probe accuracies are reported under 5-fold stratified cross-validation
with fixed random seed (42). Minimum 2 samples per class per fold required;
models with insufficient samples per class are reported separately.

### 3.5.2 Permutation test

To verify that probe accuracy reflects genuine correspondence between residual
states and reasoning types rather than classifier capacity or label artifacts,
we shuffle reasoning type labels and retrain probes on 100 independent
permutations. The $p$-value is the fraction of permutations achieving accuracy
≥ the real accuracy.

### 3.5.3 Random hidden states

We replace residual stream activations with Gaussian noise ($\mathcal{N}(0,1)$,
matching dimensionality) and retrain probes. This tests whether the probe
architecture alone can extract signal from unstructured data.

### 3.5.4 Token-count baseline

We train probes using only the number of tokens in the prompt as a feature
(padded to match residual stream dimensionality with random dimensions).
This tests whether prompt length — which varies systematically with reasoning
type — can explain probe performance.

### 3.5.5 Per-layer and per-component probes

We train separate probes on each layer's residual stream and each layer's MLP
output to localize where reasoning state information is encoded. Comparison
of residual versus MLP probe accuracy indicates whether reasoning state is
primarily carried in the residual stream or concentrated in MLP activations.

---

## 3.6 Scale-Aware Calibration

Fixed equal weights ($\alpha = \beta = \gamma = \delta = 0.25$) assume that all
mechanism stages contribute equally regardless of model scale. We calibrate
scale-aware weights using empirically observed relationships:

- $\alpha$ (${R}_{QK}$ weight): proportional to the within-model ${R}_{QK}$
  gap between direct and misleading evidence, reflecting the strength of
  evidence routing discrimination at that scale.

- $\beta$ (${M}_{AV}$ weight): proportional to the direct-evidence ${M}_{AV}$
  magnitude, reflecting how strongly evidence value vectors project to the
  answer at that scale.

- $\gamma$ (${S}_{X}$ weight): proportional to model depth, reflecting the
  observation that deeper models encode more reasoning state information.

- $\delta$ (${C}_{do}$ weight): constant baseline with small depth bonus,
  reflecting that causal sensitivity is a universal validation criterion.

Weights are normalized to sum to 1.0. The calibration is a heuristic to reveal
scale-dependent mechanism shifts; it is not claimed to be optimal. We report
per-component scores alongside any aggregate ICI.

---

## 3.7 Reproducibility

All experiments use fixed random seed 42. Models are standard HuggingFace
checkpoints loaded in evaluation mode with `output_attentions=True` and
`output_hidden_states=True`. No fine-tuning or training is performed on the
Transformer models themselves (only linear probes are trained). Experiments
were run on Apple M-series CPU. Code, data, and experiment configurations
are available at [repository].
