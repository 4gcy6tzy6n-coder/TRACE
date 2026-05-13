# Paper Outline: A Mechanistic Pathway for Evidence-to-Answer Computation in Transformers

---

## Title

**Attention Routes Evidence, but MLPs Compute Answers in Transformers**

Alternative (more conservative):
*A Mechanistic Pathway for Evidence-to-Answer Computation in Transformers*

---

## Abstract Skeleton

> Transformer reasoning is often attributed to attention, yet whether attention actually
> computes answers or merely routes evidence remains unclear. We investigate the internal
> pathway through which Transformers convert evidence into answers by systematically
> decomposing the forward pass. We find a staged mechanism: **(1)** QK attention routes
> evidence to relevant token positions, with direct-evidence samples showing 8.5–19.3×
> stronger routing than misleading-hint samples across GPT-2 and Qwen2.5 architectures.
> **(2)** MLP layers — not attention head projections — are the dominant causal pathway
> for evidence-to-answer transformation, causing 1.66× larger logit changes under ablation
> and recovering correct answers with 77.1% average patch recovery. **(3)** Residual
> streams encode distributed reasoning states, with linear probes achieving 80.5%
> classification accuracy across five reasoning types (random baseline 20%, permutation
> test p < 0.0001). **(4)** This mechanism shifts systematically with model depth: shallow
> models rely on direct attention routing, while deeper models increasingly depend on
> residual-state encoding. **(5)** Visible chain-of-thought amplifies internal state
> separability but cannot repair misleading evidence pathways, suggesting visible CoT is
> a partial externalization rather than the mechanism itself. To quantify this pathway,
> we introduce the Internal CoT Index (ICI), decomposable into QK routing, MLP-mediated
> message contribution, residual-state separability, and causal intervention sensitivity.
> These findings provide a structural skeleton for understanding how Transformers
> internally produce answers from evidence, independent of whether they generate
> explicit reasoning traces.

---

## 1. Introduction

### 1.1 Motivation
- Chain-of-thought is widely used but its relationship to internal computation is unclear
- Attention is often treated as the primary explanation mechanism
- Need: a mechanistic account of *how* evidence becomes answer internally

### 1.2 Core Question
How do Transformers internally convert evidence into answers?

### 1.3 Approach
- Systematic decomposition of Transformer internal variables (Q, K, V, A, W_O, W_U, X_l, MLP)
- Four mechanism stages: routing → transformation → storage → output
- ICI as quantification tool, not the primary contribution

### 1.4 Contributions (4-point structure)
1. Mechanism chain: QK → MLP → X_l → logits
2. Attention/MLP division of labor: attention routes, MLP computes
3. Residual reasoning state: probe validation with permutation controls
4. Visible CoT boundary: amplifier, not mechanism

---

## 2. Related Work

### 2.1 Chain-of-Thought and Faithfulness
- CoT improves accuracy but faithfulness is debated
- Our contribution: measure internal mechanism directly, not external text

### 2.2 Mechanistic Interpretability
- Circuits-style: identifies specific circuits for narrow tasks
- Our contribution: general structural skeleton across tasks

### 2.3 Attention as Explanation
- Attention weights correlate with importance
- Our contribution: attention routes, but does not compute answers

### 2.4 Probing and Representation Analysis
- Classifiers on hidden states
- Our contribution: permutation tests, token-count baselines, cross-architecture

### 2.5 Activation Patching and Causal Analysis
- Resample ablation, causal tracing
- Our contribution: multi-component comparison (QK, attn, MLP, residual) with strict controls

---

## 3. Method: Decomposing the Evidence-to-Answer Pathway

### 3.1 Model and Data
- Models: GPT-2 family (82M–355M), Qwen2.5-0.5B, Pythia
- Data: 200 controlled reasoning samples, 5 types

### 3.2 Internal Variable Extraction
- Q/K/V: fused c_attn hook → split → per-head reshaping
- Verification: softmax(QK^T/√d) vs model attention (tol 10^{-5})

### 3.3 Four Mechanism Stages

**Stage 1: QK Evidence Routing (R_QK)**
- R_QK = attention mass from answer to evidence positions
- Per-layer, per-head decomposition

**Stage 2: MLP Evidence Transformation (M_AV)**
- True M_AV: A × V × W_O(head) × W_U[answer]
- Distributed M_AV: attention pathway + MLP pathway + residual pathway

**Stage 3: Residual Reasoning State (S_X)**
- Linear probe on residual stream
- 5-class classification with permutation test

**Stage 4: Causal Intervention (C_do)**
- Component-specific ablation (QK, attention, MLP, residual)
- Activation patching with 6 control types
- MLP-specific causal test

### 3.4 ICI Aggregation
- ICI = αR_QK + βM_AV + γS_X + δC_do
- Scale-aware weight calibration

---

## 4. Experiments

### 4.1 QK Routes Evidence
- **Main result**: R_QK(direct) / R_QK(misleading) = 8.5–19.3×
- Cross-architecture: GPT-2 and Qwen2.5
- QK reconstruction verified: all 12 layers pass
- Early-layer concentration: evidence routing peaks in layers 0-1

### 4.2 MLP Computes Answers
- **Main result**: MLP |Δlogit| = 98.1 vs attention = 59.3 (1.66×)
- 10/12 layers MLP-dominant by causal effect
- MLP patch recovery: 77.1% average
- Distributed pathway: MLP fraction 13-43% vs attention 4-7%

### 4.3 Residual Stores Reasoning State
- **Main result**: S_X = 0.756 (200 samples), p < 0.0001
- Permutation test: shuffled = 20.4%, random states = 15.0% (S_X = 0.000)
- Token-count baseline: 41.0% (S_X = 0.262)
- S_X increases with model depth: 0.583 (6L) → 0.542 (12L) → 0.708 (24L)

### 4.4 Mechanism Shifts with Scale
- **Main result**: α↓ (0.306→0.197), γ↑ (0.249→0.445)
- Weight shift: routing-dominant (6L) → state-encoding-dominant (24L)
- MLP pathway fraction peaks at 12L (42.8%)

### 4.5 Visible CoT Is Not the Mechanism
- **Main result**: CoT boosts S_X (+25%) but cannot fix misleading (ΔICI = 0.000)
- Faithful vs unfaithful CoT: identical ICI in GPT-2 small
- Implication: visible CoT amplifies existing mechanisms, not creates them

---

## 5. Discussion

### 5.1 What We Found
- Structural skeleton of evidence-to-answer computation
- Division of labor: attention routes, MLP computes
- Residual stream as distributed reasoning workspace

### 5.2 What We Do Not Claim
- Full reverse-engineering of Transformer reasoning
- MLP feature-level mechanism details
- Universality across all model scales (tested 82M–494M)

### 5.3 Implications
- **Diagnosis**: misleading prompts leave measurable internal traces
- **Architecture**: strengthen MLP, not just attention, for reasoning
- **Training**: S_X as potential internal reasoning objective
- **CoT debate**: visible CoT as amplifier, not mechanism

### 5.4 Limitations
- Model scale range: 82M–494M
- Task diversity: controlled synthetic, not real-world QA
- Two-phase MLP: observed but not fully validated
- Causal completeness: full mediation pending

---

## 6. Conclusion

Transformers convert evidence to answers through a staged internal mechanism:
QK attention routes evidence to relevant positions, MLP layers causally transform
this evidence into answer-relevant reasoning states, residual streams store
distributed intermediate representations, and logits produce final answers.
Visible chain-of-thought is a partial externalization of this internal mechanism,
not the mechanism itself. These findings provide a structural skeleton for
understanding internal reasoning in Transformers, with implications for model
diagnosis, architecture design, and the interpretation of chain-of-thought.

---

## Appendix: Reproducibility

- All models: standard HuggingFace checkpoints
- All code: open-source pipeline at [repository]
- Random seeds: fixed at 42
- Hardware: CPU (Apple M-series) for all experiments
- Data: 200 controlled samples included
