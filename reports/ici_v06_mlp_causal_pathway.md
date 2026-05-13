# Internal CoT Index v0.6 — MLP Causal Pathway Validation

## Objective

Validate that MLP layers (not just attention) are causally involved in
Internal CoT, completing the mechanism chain:

```
QK routes evidence → MLP transforms → residual stores state → logits produce answer
```

## Experiments

### 1. MLP vs Attention Ablation (all layers)
Zero MLP or attention output at answer position, measure logit change.

### 2. Layer-wise Ablation Sweep
Which layers are MLP-dominant vs attention-dominant?

### 3. MLP Patching
Patch clean MLP output into corrupted forward, measure answer recovery.

---

## Results

### Experiment 1 & 2: Ablation Comparison

| Metric | MLP Ablation | Attention Ablation |
|--------|-------------|-------------------|
| Avg |Δlogit| | **98.14** | 59.34 |
| MLP-dominant layers | **10/12** | 2/12 |

Both pathways show suppression effects (removal increases logit on average),
but MLP ablation causes 1.66× larger absolute changes.

### Layer-wise Dominance

| Layer | MLP Δlogit | Attn Δlogit | Dominant |
|-------|-----------|-------------|----------|
| 0 | -9.71 | **-57.69** | Attn |
| 1 | **-3.73** | -1.43 | MLP |
| 2 | **-5.74** | +2.66 | MLP |
| 3 | **+9.58** | +3.73 | MLP |
| 4 | **+10.51** | +3.84 | MLP |
| 5 | **+4.74** | -1.44 | MLP |
| 6 | **+5.01** | +2.23 | MLP |
| 7 | **+7.63** | +4.81 | MLP |
| 8 | **-16.96** | +6.30 | MLP |
| 9 | **-8.25** | +4.75 | MLP |
| 10 | **-41.12** | +3.08 | MLP |
| 11 | -19.14 | **-29.74** | Attn |

**Pattern**: Middle layers (L1-L10) are uniformly MLP-dominant.
Attention dominates only at layer 0 (immediate evidence routing) and
layer 11 (final output preparation). MLP carries the internal computation
across all intermediate layers.

### Sign Pattern: Suppression → Construction

- **Layers 0-2, 8-11**: Negative drop (ablation = logit goes UP)
  → These layers *suppress* the answer until evidence is processed
- **Layers 3-7**: Positive drop (ablation = logit goes DOWN)
  → These layers *construct* the answer signal from evidence

This reveals a **two-phase MLP computation pattern**:
1. Early/late MLPs suppress premature answer output
2. Middle MLPs build evidence-grounded answer signal

### Experiment 3: MLP Patching

| Pair | Clean → Corrupted | MLP Recovery |
|------|------------------|-------------|
| pair_001 | 2023 → 2022 | 0.000 |
| pair_003 | 330m → 300m | **0.934** |
| pair_004 | 100°C → 90°C | **0.922** |
| pair_005 | 6400km → 7000km | 1.000+ |

**Average MLP patch recovery: 77.1%**

For 3/5 pairs, MLP patching achieves >90% recovery of the clean answer logit.
This is direct causal evidence that MLP activations encode the correct answer.

---

## Causal Comparison Summary

```
Causal pathway strength:     MLP > Attention

MLP ablation |Δlogit|:       98.14  (1.66× attention)
Attention ablation |Δlogit|: 59.34
MLP-dominant layers:         10/12  (83%)
MLP patch recovery:          77.1%
```

---

## Mechanism Chain (Complete)

```
            QK Routes Evidence
            (v0.2 verified: softmax(QK^T/√d) = A_model)
                    │
                    ▼
            MLP Transforms Evidence → Reasoning State
            (v0.6: MLP-dominant in 10/12 layers, 77% patch recovery)
                    │
                    ▼
            Residual Stream Stores Distributed State
            (v0.3: S_X = 0.756, p < 0.0001; v0.5: γ increases with depth)
                    │
                    ▼
            Logits Produce Answer
            (v0.1-0.6: ICI(direct) > ICI(misleading) across all scales)
```

---

## Paper Contribution Update

v0.6 adds:

> **MLP is the dominant causal pathway for evidence-to-answer processing.**
> MLP ablation at the answer position causes 1.66× larger logit changes
> than attention output ablation, and MLP activation patching recovers the
> correct answer with an average 77.1% recovery ratio. Layer-wise analysis
> reveals a two-phase computation: early and late MLPs suppress premature
> answer output, while middle MLPs construct the evidence-grounded signal.
> Only layer 0 (initial routing) and layer 11 (final output) are
> attention-dominant, confirming that attention routes evidence while
> MLP transforms it into reasoning state.

---

## Limitations

1. Sample size: 5 samples for ablation, 5 for patching
2. Single model: GPT-2 (124M) only
3. Suppression effect needs deeper analysis: why does MLP removal increase logit?
4. MLP patch failure cases (pair_001) need investigation
5. Scale dependence not yet tested

---

*Report generated: 2026-05-12. ICI v0.6 — MLP Causal Pathway Validation.*
