# 5. Discussion

*Draft — May 2026*

---

We have shown that Transformer reasoning proceeds through a staged internal
mechanism — QK routes evidence, MLP transforms it, residual streams store
intermediate states — and that this mechanism can be operationalized to reduce
black-box failures. Here we discuss what these findings mean, what they do not
mean, and where they point.

---

## 5.1 From Explaining to Improving

The dominant approaches to Transformer interpretability — attention visualization,
probing, and chain-of-thought analysis — share a common limitation: they explain
or describe model behavior without directly changing it. Our findings suggest a
different relationship between understanding and improvement. The same internal
variables that reveal the evidence-to-answer pathway can be used to intervene on
that pathway when it weakens.

This matters for practical deployment. Current reliability techniques —
confidence thresholding, self-consistency, retrieval augmentation — operate on
external signals (output probabilities, multiple samples, external knowledge).
TRACE operates on internal mechanism signals, enabling intervention that is
matched to the specific type of mechanism weakness: abstention when evidence is
insufficient, conflict disclosure when evidence is contradictory, and routing
repair when evidence is misdirected. The >1B-scale results (§4.7) demonstrate
that this approach is not merely diagnostic but can reduce actual failures.

---

## 5.2 The Division of Labor Between Attention and MLP

A recurring theme in our results is that attention and MLP serve different
roles in the evidence-to-answer pathway. QK attention routes evidence to
relevant positions — it identifies *where* information is. MLP layers
causally transform this information into answer-relevant states — they
determine *what* is done with it.

This division of labor has implications beyond the specific mechanism we
study. It suggests that attention-centric interpretability methods, which
focus on attention weights as the primary explanatory lens, may be
systematically underestimating the role of MLP computation. Our finding
that MLP ablation causes 1.4–3.0× larger logit changes than attention
ablation, across 54 of 54 tested samples and five task formats, indicates
that the MLP's role in answer computation has been underappreciated.

It also suggests architectural implications. If MLP is the primary
evidence-to-answer transformation pathway, then efforts to improve
reasoning fidelity — through architecture design, training objectives,
or inference-time intervention — should target MLP computation at least
as much as attention patterns.

---

## 5.3 The Boundary at Misleading Evidence

The most consistent limitation across our experiments is TRACE's partial
effectiveness on misleading-hint samples. While conflict and evidence-gap
failures are nearly eliminated by mechanism-matched intervention (§4.7),
misleading-driven errors are reduced by only 22% on the 1.5B model and
0% on the 3B model.

This is not a measurement failure; it is a mechanism boundary. Misleading
evidence operates by hijacking the QK routing mechanism itself: the model
attends to evidence tokens that genuinely support the misleading
interpretation, and filtering a few cue phrases from the prompt is
insufficient to redirect this routing. The remaining evidence — even when
it contradicts the misleading claim — may not be structured in a way that
QK routing can isolate as the primary evidence source.

This boundary has practical significance. It means that prompt-level
intervention cannot fully compensate for evidence that is structurally
misleading. In deployment settings, misleading-evidence detection may
require deeper intervention — retraining with adversarial examples,
architectural modifications to routing mechanisms, or external fact
verification rather than internal trace repair alone.

---

## 5.4 Scale Dependence

Our experiments span 82M to 3B parameters. Three findings show
scale-dependent patterns:

First, residual state encoding (S_X) increases with model depth,
from 0.58 (6L) to 0.71 (24L) in the GPT-2 family. Deeper models
encode more reasoning-state information in their residual streams.

Second, MLP dominance is strongest at intermediate depth (12L,
1.70× over attention) and converges with attention at greater depth
(24L, 0.76×). This suggests that evidence-to-answer computation
becomes more distributed — across attention and MLP pathways — as
models deepen, rather than concentrating in a single dominant pathway.

Third, error reduction from TRACE intervention is observed at both
1.5B (75%) and 3B (57%), with no evidence of diminishment at the
larger scale. Whether these patterns continue to larger models
(7B, 70B+) is an open question that our measurement infrastructure
is designed to test.

---

## 5.5 What We Do Not Claim

We do not claim to have fully reverse-engineered Transformer reasoning.
Our contribution is the identification of a structural skeleton —
QK routing → MLP transformation → residual state storage → logit
projection — and the demonstration that this skeleton can be measured,
decomposed, causally validated, and operationalized for intervention.
The detailed feature-level computation within each stage, particularly
the specific MLP features that mediate the transition from evidence
routing to answer construction, remains for future work.

We do not claim that TRACE eliminates all black-box failures. The
misleading-evidence boundary (§5.3) is a documented limitation, and
real-world deployment would involve failure modes beyond the five
reasoning types studied here.

We do not claim that our specific intervention strategies — conservative
prompting, cue filtering, evidence reformatting — are optimal. They are
first demonstrations that mechanism-matched intervention can work.
Optimization of intervention strategies, automated intervention selection,
and integration with existing reliability techniques are natural next steps.

We do not claim that the mechanism chain is universal across all
Transformer architectures and training paradigms. We have validated it
on GPT-2 and Qwen2.5 families (82M–3B parameters) across controlled
reasoning and fact-verification tasks. Generalization to encoder-decoder
architectures, mixture-of-experts models, and models trained with
reinforcement learning remains to be tested.

---

## 5.6 Implications

**For interpretability research.** Our results suggest that the field's
emphasis on attention as the primary explanatory mechanism may need
rebalancing toward MLP computation and residual stream state analysis.
The tools we introduce — QK reconstruction verification, per-component
causal ablation, distributed pathway decomposition, and mechanism-matched
intervention — provide a template for this rebalancing.

**For model deployment.** TRACE demonstrates that internal mechanism
traces can serve as reliability signals that are more selective than
confidence scores and more mechanism-grounded than attention patterns.
For high-stakes applications where unsupported answers and conflict
non-disclosure are unacceptable, mechanism-grounded auditing offers a
new category of safeguard.

**For the chain-of-thought debate.** Our findings support a specific
relationship between visible CoT and internal mechanism: CoT amplifies
existing internal pathways but cannot create them where they are absent
or repair them when they are misdirected. This suggests that CoT
faithfulness research should attend not only to whether CoT text matches
model behavior, but to whether the underlying internal mechanism chain
is intact — a measurement that TRACE enables.

**For architecture design.** If MLP is the primary evidence-to-answer
transformation pathway, architectures that strengthen MLP computation
for reasoning tasks — deeper MLPs, gated MLP pathways, or MLP-focused
training objectives — may improve reasoning fidelity more efficiently
than attention modifications alone.

---

## 5.7 Open Questions

1. **MLP feature-level mechanism.** Which specific MLP features mediate
   the transition from evidence routing to answer construction? Our
   two-phase MLP observation (early/late suppression, middle construction)
   in GPT-2 12L requires feature-level validation.

2. **Scale generalizability.** Do the mechanism chain and TRACE error
   reduction hold at 7B, 70B, and beyond? The measurement infrastructure
   is architecture-agnostic; model access is the primary constraint.

3. **Training-time integration.** Can TRACE-style internal auditing be
   incorporated into training objectives, so that models learn to
   maintain strong internal evidence-to-answer pathways rather than
   relying on post-hoc intervention?

4. **Real-world deployment.** How does TRACE perform on diverse real-world
   tasks beyond controlled reasoning and fact verification? Integration
   with existing retrieval-augmented generation and factuality evaluation
   pipelines is a natural extension.

5. **Human-AI interaction.** Can human auditors use internal evidence-to-answer
   traces to make better-calibrated trust decisions than they can with
   external CoT or confidence scores alone?
