# 2. Related Work

*Draft — May 2026*

---

Our work intersects five research areas. For each, we clarify how our
findings extend, complement, or challenge existing understanding.

## 2.1 Chain-of-Thought and Reasoning Faithfulness

Chain-of-thought prompting improves model accuracy on reasoning tasks [1, 2],
but the relationship between visible CoT and internal computation remains
debated. Studies have found that CoT can be unfaithful — models may produce
plausible reasoning traces that do not reflect their actual computation [3, 4].
Our findings provide a mechanistic perspective on this debate: visible CoT
amplifies existing internal evidence-to-answer pathways (§4.5) but cannot
create pathways where none exist. When evidence routing is systematically
misdirected by misleading prompts, CoT has zero effect on internal mechanism
strength. This suggests that CoT faithfulness depends on whether the
underlying mechanism chain is intact — a condition that external CoT
analysis alone cannot verify.

## 2.2 Mechanistic Interpretability

Mechanistic interpretability aims to reverse-engineer neural network
computation into human-understandable components [5, 6]. Most work in this
tradition identifies specific circuits for narrow tasks — induction heads [7],
IOI circuits [8], factual association pathways [9]. Our work differs in
identifying a general structural skeleton (QK routing → MLP transformation →
residual storage → logit projection) that applies across reasoning types
rather than to a single task. This skeleton is coarser than feature-level
circuits but broader in applicability: it describes *what kind* of
computation occurs at each stage, rather than *which specific features*
implement it. The two levels are complementary: circuits research can fill
in feature-level detail within the structural skeleton we identify.

## 2.3 Attention as Explanation

The debate over whether attention weights provide meaningful explanations [10,
11] has largely focused on whether attention correlates with model behavior.
Our results suggest a different framing: attention does explain something —
evidence routing — but it does not explain the transformation of evidence into
answers. QK routing discriminates direct from misleading evidence by 7.6–19.3×
(§4.1), and attention entropy is completely non-predictive of answer errors
(§4.7). This division of explanatory labor — attention explains routing,
MLP explains transformation — resolves the apparent contradiction in prior
work: attention *is* explanatory for the specific computational role it serves,
but that role is narrower than the full reasoning process.

## 2.4 Probing and Representation Analysis

Linear probes on hidden states have been used to detect linguistic features,
factual knowledge, and reasoning intermediates [12, 13]. Our residual state
probe results (S_X = 0.50–0.99, §4.3) extend this tradition with three
controls that rule out common confounds: permutation tests (p < 0.0001),
random hidden states (S_X = 0.000), and token-count baselines (S_X = 0.262).
These controls demonstrate that residual streams encode reasoning-type
information beyond what surface-level features predict. Our layer-wise probe
results further show that this information is distributed across all layers
rather than concentrated in specific layers — consistent with the residual
stream architecture.

## 2.5 Activation Patching and Causal Analysis

Activation patching [9, 14] tests whether specific activations are causally
necessary for model outputs. Our patching experiments extend this methodology
in two ways: (1) we compare attention-output patching, MLP-output patching,
and full residual patching to decompose causal contributions by component
(§4.2), and (2) we implement six control conditions (correct, random,
unrelated, same-type-wrong, evidence-only, answer-only) to rule out
confounds from distribution shift or position artifacts (§3.4). The
finding that MLP patching recovers correct answers with 77.1% average
recovery, and that MLP ablation causes 1.4–3.0× larger effects than
attention ablation, provides causal evidence for the MLP's dominant role
in evidence-to-answer transformation.

## 2.6 Black-Box Reliability and Failure Reduction

Recent work on reducing language model failures has focused on
confidence-based abstention [15], self-consistency [2], retrieval
augmentation [16], and reinforcement learning from human feedback [17].
These methods operate on external signals: output probabilities, multiple
samples, retrieved documents, or human preferences. TRACE introduces a
different category of reliability signal: internal mechanism traces.
Our >1B-scale results (§4.7) demonstrate that mechanism-grounded
intervention can reduce failures by 57–75% without increasing false
positives — a level of reduction comparable to or exceeding
confidence-based approaches, with the additional property of being
selective (intervention only when the mechanism indicates weakness)
rather than uniform (intervention whenever confidence is low).

---

## References (Preliminary)

[1] Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in LLMs.* NeurIPS 2022.
[2] Wang et al. *Self-Consistency Improves CoT Reasoning.* ICLR 2023.
[3] Turpin et al. *Language Models Don't Always Say What They Think.* NeurIPS 2024.
[4] Anthropic. *Studying Chain-of-Thought Faithfulness.* 2024.
[5] Olah et al. *The Building Blocks of Interpretability.* Distill 2018.
[6] Elhage et al. *A Mathematical Framework for Transformer Circuits.* 2021.
[7] Olsson et al. *In-Context Learning and Induction Heads.* 2022.
[8] Wang et al. *Interpretability in the Wild: a Benchmark for Lying-refusing Circuits.* 2022.
[9] Meng et al. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
[10] Jain & Wallace. *Attention is not Explanation.* NAACL 2019.
[11] Wiegreffe & Pinter. *Attention is not not Explanation.* EMNLP 2019.
[12] Alain & Bengio. *Understanding Intermediate Layers Using Linear Classifier Probes.* 2017.
[13] Hewitt & Manning. *A Structural Probe for Finding Syntax in Word Representations.* NAACL 2019.
[14] Vig et al. *Causal Mediation Analysis for Interpreting Neural NLP.* EMNLP 2020.
[15] Kadavath et al. *Language Models (Mostly) Know What They Know.* 2022.
[16] Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
[17] Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022.
