# Nature Storyline

*Frozen: May 2026*

---

## Problem

Transformers produce answers from evidence, but the internal pathway from
evidence to answer is opaque. Attention weights show where information may
flow. Chain-of-thought shows what a model says it is doing. Neither explains
how evidence is internally converted into an answer. Without this
understanding, we cannot distinguish faithful reasoning from plausible
rationalization, diagnose why models fail on misleading prompts, or
design architectures that strengthen genuine evidence integration.

## Discovery

We find that evidence-to-answer processing follows a staged internal pathway:

```
QK attention routes evidence → MLP transforms evidence → residual stream stores reasoning state → logits produce answer
```

- **QK routes evidence**: Attention mass to evidence tokens is 8.5–19.3× higher
  for direct-evidence than misleading-hint samples, across GPT-2 and Qwen2.5
  architectures. QK reconstruction verified: softmax(QK^T/√d) matches model
  attention in all 12 layers (tol 10^{-5}).

- **MLP transforms evidence**: MLP output ablation causes 1.66× larger logit
  changes than attention output ablation. MLP is dominant in 10/12 layers.
  MLP activation patching recovers correct answers with 77.1% recovery.
  MLP > attention in 51/51 cross-format samples.

- **Residual stream stores reasoning state**: Linear probes achieve 80.5%
  accuracy classifying five reasoning types from residual activations
  (S_X = 0.756, permutation test p < 0.0001). Random states: S_X = 0.000.
  Token-count baseline: S_X = 0.262.

## Measurement

We introduce the Internal Causal Index (ICI) to quantify the mechanism chain:
ICI = αR_QK + βM_AV + γS_X + δC_do. ICI is the measurement tool; the
mechanism chain is the discovery.

## Audit & Intervention

We convert the mechanism discovery into TRACE (Transformer Reasoning Auditing
through Causal Evidence), a framework that extracts internal evidence-to-answer
traces, diagnoses mechanism weaknesses, and applies matched interventions.

- **Audit**: TRACE flags 88.9% of misleading-hint samples as risky while
  flagging 0% of direct-evidence samples — higher precision than confidence
  scores (0.455 vs 0.370).

- **Intervention (Oracle)**: In controlled mechanism-matched intervention, TRACE
  reduces black-box failures by 70.3% [95% CI: 61.1, 78.8] on Qwen2.5-1.5B (>1B),
  sharply reducing conflict non-disclosure (97.4% reduction, p < 0.0001) and
  evidence-gap unsupported answers (95.0%, p < 0.0001). Direct-evidence false
  positives show no increase (2/40 in both raw and TRACE).

- **Cross-architecture**: Pattern replicates on Qwen2.5-3B (57.2% reduction)
  and LLaMA-3.2-1B-Instruct (22.1% [12.8, 32.1], p = 0.00024).

## Intervention Is Not Prompt Engineering

Intervention ablation proves the effect is mechanism-grounded, not a prompt
artifact. Mismatched intervention (wrong type) is 1.9× worse than matched
(p = 0.0013). Random intervention is 1.5× worse (p = 0.0212). No intervention
restores the full error rate (97.5%, p < 0.0001).

## Autonomous Trigger

TRACE V3.1 operates without gold reasoning-type labels, using only internal
signals (R_QK + confidence + residual-state probe). It achieves 8.8% error
rate [4.3, 17.0] with 71% fire rate — a 23% reduction in unnecessary
intervention compared to simple trace-only triggering (92% fire rate).
Further fire-rate reduction (V3.2, 61% fire) increases error to 18.8%,
revealing a safety–utility calibration frontier.

## Mechanism Boundary: Misleading Evidence

Misleading-hint failures show a distinct internal mechanism: misleading cue
tokens hijack QK routing (R_QK = 0.45–0.51) while gold evidence receives
13–108× lower attention (R_QK = 0.005–0.015). This is fundamentally different
from conflict, where attention distributes across both evidence spans.
Prompt-level filtering cannot redirect hijacked routing, identifying a
mechanism boundary for current TRACE intervention.

## Contributions

1. **Mechanism chain**: QK routes evidence → MLP transforms → residual stores
   state → logits produce answer. Validated across 3 architectures and 5 formats.

2. **MLP dominance**: MLP is the primary causal pathway for evidence-to-answer
   transformation (1.4–3.0× over attention, 51/51 cross-format samples).

3. **Residual state**: Residual streams encode reasoning states above all
   measured confounds (S_X = 0.50–0.99, all p < 0.0001).

4. **TRACE**: In controlled mechanism-matched interventions, TRACE reduces
   black-box failures by 57–75% on >1B models without increasing direct-evidence
   false positives. TRACE V3.1 achieves gold-label-free autonomous triggering.

5. **Autonomous operation**: TRACE V3.1 operates without gold labels,
   demonstrating that internal mechanism signals can drive selective
   intervention.

## Scope and Boundaries

We do not claim to have fully reverse-engineered Transformer reasoning.
We identify a structural skeleton for evidence-to-answer computation.
Misleading-driven errors remain partially resistant. Autonomous trigger
calibration presents a safety–utility trade-off at current signal resolution.
These boundaries are documented, not hidden.

## One-Sentence Thesis

> Transformer reasoning proceeds through a staged internal mechanism — QK
> routes evidence, MLP transforms it, residual streams store intermediate
> states — that can be measured, causally validated, and operationalized
> to reduce black-box failures without relying on external chain-of-thought.
