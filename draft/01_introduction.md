# 1. Introduction

*Draft — May 2026*

---

## Opening

Attention tells us where information may flow in a Transformer.
Chain-of-thought tells us what a model says it is doing.
Neither alone explains how a Transformer internally converts evidence into an answer.

This gap matters. If we cannot trace the internal pathway from evidence to answer,
we cannot distinguish faithful reasoning from plausible-sounding rationalization.
We cannot diagnose why a model fails on misleading prompts even when it produces
correct-looking chain-of-thought. We cannot design architectures that strengthen
genuine evidence integration rather than surface-level pattern matching.

---

## The Problem

The dominant paradigm for interpreting Transformer reasoning operates at two levels.
At the **architectural level**, attention weights are treated as the primary explanatory
lens — high attention to a token is interpreted as the model "using" that token [1–3].
At the **behavioral level**, chain-of-thought prompting is used to elicit explicit
reasoning traces, which are then evaluated for faithfulness and consistency [4–6].

Both approaches have advanced our understanding, but both face a fundamental limitation:
they observe external correlates of reasoning — attention patterns or generated text —
rather than the internal computation that actually produces answers from evidence.

This limitation has concrete consequences. Attention can highlight evidence without
the model causally depending on it [7, 8]. Chain-of-thought can appear faithful while
masking internally inconsistent reasoning [9, 10]. And when chain-of-thought is absent
or unreliable, we have no way to assess whether a model's answer is grounded in
evidence or produced through shallow heuristics.

---

## Our Question

We ask a more basic question:

> **How do Transformers internally convert evidence into answers?**

This question is not about whether attention is interpretable, or whether chain-of-thought
is faithful. It is about identifying the internal computational pathway — the sequence
of operations across specific Transformer components — that transforms input evidence
into output answers.

Answering this question requires moving beyond both attention visualization and
external text analysis. It requires decomposing the forward pass into its constituent
operations — query, key, value projections, attention weight computation, multi-layer
perceptron transformations, residual stream accumulation, and logit projection — and
measuring how each stage contributes to the evidence-to-answer pathway.

---

## Our Approach

We systematically decompose the Transformer forward pass to trace evidence-to-answer
computation through four stages:

**Stage 1: QK Attention — Evidence Routing.**
We measure whether attention weights route information from evidence-bearing tokens
to answer-relevant positions. We extract per-head Q, K, V projections, verify that
recomputed attention matches model attention ($\text{softmax}(QK^T/\sqrt{d}) \approx A$,
tolerance $10^{-5}$), and quantify evidence routing strength ($R_{QK}$) as the attention
mass from answer positions to evidence tokens.

**Stage 2: MLP Layers — Evidence Transformation.**
We measure whether evidence information is causally transformed through MLP layers
into answer-relevant signals. We compute the contribution of evidence value vectors
through attention head projections to the answer logit ($M_{AV}$, the $A \times V
\times W_O \times W_U$ pathway), decompose this into attention-pathway and
MLP-pathway components, and perform component-specific causal interventions
(ablation and activation patching) to establish which pathway dominates.

**Stage 3: Residual Stream — Reasoning State Storage.**
We measure whether residual streams encode intermediate reasoning states beyond what
surface-level features would predict. We train linear probes to classify reasoning
types (direct evidence, conflict, evidence gap, misleading hint, multi-step) from
residual stream activations, and validate probe accuracy against three controls:
label permutation, random hidden states, and token-count baselines.

**Stage 4: Logit Projection — Answer Production.**
We measure whether the accumulated internal computation causally determines the
final answer by intervening on internal pathways (token ablation, attention masking,
component-specific ablation, activation patching with six control types) and
quantifying the resulting logit change ($C_{do}$).

To integrate these four measurements, we introduce the Internal CoT Index (ICI),
a composite score that quantifies the strength of the evidence-to-answer pathway:
$ICI = \alpha R_{QK} + \beta M_{AV} + \gamma S_X + \delta C_{do}$, with scale-aware
weight calibration. ICI is not our primary contribution; it is the measurement tool
that enables systematic comparison across models, tasks, and conditions.

---

## Main Findings

We evaluate this framework on 200 controlled reasoning samples spanning five types
(direct evidence, conflict, evidence gap, misleading hint, multi-step reasoning)
across GPT-2 (82M–355M) and Qwen2.5 (494M) model families. Our main findings are:

**1. QK attention routes evidence, but does not compute answers.**
Evidence routing strength ($R_{QK}$) is 8.5–19.3× higher for direct-evidence samples
than for misleading-hint samples. This discrimination holds across GPT-2 and Qwen2.5
architectures. However, routing alone does not determine the answer — high $R_{QK}$
can coexist with incorrect output when evidence is routed but not causally processed.

**2. MLP layers — not attention head projections — are the dominant causal pathway
for evidence-to-answer transformation.**
MLP output ablation causes 1.66× larger absolute logit changes than attention output
ablation (|Δlogit| = 98.1 vs. 59.3), and MLP activation patching recovers correct
answers with 77.1% average recovery from corrupted evidence. Layer-wise analysis
reveals that 10 of 12 GPT-2 layers are MLP-dominant by causal effect size, with
only the first and last layers showing attention dominance. This suggests a division
of labor: attention finds evidence, MLPs compute with it.

**3. Residual streams encode distributed reasoning states above all measured confounds.**
Linear probes achieve 80.5% accuracy in classifying reasoning type from residual
stream activations ($S_X = 0.756$, five-class random baseline 20%). Permutation
tests confirm this is not a label artifact ($p < 0.0001$, 100 shuffles). Random
hidden states yield $S_X = 0.000$. A token-count baseline — which captures prompt
length variation across reasoning types — achieves only 41.0% accuracy ($S_X = 0.262$),
demonstrating that residual streams encode reasoning-specific information beyond
surface-level features.

**4. The internal mechanism shifts systematically with model depth.**
As models deepen from 6 to 24 layers, scale-aware ICI weights shift from
routing-dominant ($\alpha = 0.306$, 6L) to state-encoding-dominant ($\gamma = 0.445$,
24L). MLP-mediated evidence processing peaks at intermediate scale (12 layers,
42.8% MLP pathway fraction). This suggests that internal reasoning is not one
mechanism but a collection of mechanisms whose relative importance changes with
model capacity — a finding that fixed-weight composite metrics would mask.

**5. Visible chain-of-thought is an amplifier of internal mechanisms, not the
mechanism itself.**
Chain-of-thought prompting increases residual-state separability ($S_X$ rises from
0.550 to 0.575, +4.5%), and modestly increases ICI for reasoning-required types
(direct evidence: +0.016, conflict: +0.018). However, CoT has zero effect on
misleading-hint ICI (+0.000), and GPT-2 small produces identical ICI scores for
faithful and unfaithful CoT narratives. This indicates that visible CoT can amplify
existing internal evidence pathways but cannot create pathways where none exist or
repair pathways that are systematically misdirected.

---

## Contributions

1. **A mechanistic account of evidence-to-answer computation in Transformers.**
   We identify a staged internal pathway — QK routes evidence → MLP transforms
   evidence → residual streams store reasoning states → logits produce answers —
   and provide quantitative evidence for each stage.

2. **Evidence that MLP, not attention, is the dominant causal pathway for
   evidence-to-answer transformation.** This challenges the attention-centric
   view of Transformer reasoning and suggests that the MLP's role in internal
   computation has been underestimated.

3. **Validation that residual streams encode reasoning states**, with controls
   ruling out label artifacts, random correlations, and surface-level confounds.

4. **A measurement framework (Internal CoT Index) and open-source pipeline**
   that maps mechanism stages to specific Transformer variables and provides
   causal validation through ablation, patching, and permutation testing.

---

## Scope and Boundaries

We do not claim to have fully reverse-engineered Transformer reasoning.
Our contribution is the identification of a structural skeleton for
evidence-to-answer computation, and the demonstration that this skeleton
can be measured, decomposed, and causally validated. The detailed feature-level
computation within each stage — particularly the specific MLP features that
mediate the suppression-to-construction transition, and the exact nature of
distributed residual state representations — remains for future work.

Our experiments are conducted on models up to 494M parameters on controlled
reasoning tasks. Cross-architecture replication (GPT-2 and Qwen2.5) provides
initial evidence of generality, but validation on larger models and real-world
tasks is needed.

What we establish is that Transformer reasoning is not attention-only; that
MLPs play a larger causal role in evidence-to-answer transformation than
previously emphasized; and that internal reasoning traces can be quantified
without relying on external chain-of-thought text.

---

## Paper Structure

Section 2 reviews related work. Section 3 describes our method for decomposing
the evidence-to-answer pathway. Section 4 presents our main experimental results
organized by mechanism stage. Section 5 discusses implications, limitations, and
open questions. Section 6 concludes.

---

## References (Preliminary)

[1] Jain & Wallace. *Attention is not Explanation.* NAACL 2019.
[2] Wiegreffe & Pinter. *Attention is not not Explanation.* EMNLP 2019.
[3] Clark et al. *What Does BERT Look At?* ACL 2019.
[4] Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in LLMs.* NeurIPS 2022.
[5] Wang et al. *Self-Consistency Improves CoT Reasoning.* ICLR 2023.
[6] Lanham et al. *Measuring Faithfulness in CoT Reasoning.* 2023.
[7] Pruthi et al. *Learning to Deceive with Attention.* ACL 2020.
[8] Bastings et al. *Will You Find These Shortcuts?* EMNLP 2021.
[9] Turpin et al. *Language Models Don't Always Say What They Think.* NeurIPS 2024.
[10] Anthropic. *Studying Chain-of-Thought Faithfulness.* 2024.
