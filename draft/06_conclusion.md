# 6. Conclusion

*Draft — May 2026*

---

How do Transformers internally convert evidence into answers? We have shown
that this process follows a staged mechanism: QK attention routes evidence to
relevant token positions, MLP layers causally transform it into
answer-relevant reasoning states, residual streams store distributed
intermediate representations, and logits produce final answers.

This mechanism is not a single architectural property but a collection of
operations whose relative importance shifts with model depth — from
routing-dominant in shallow models to state-encoding-dominant in deeper
ones — and whose components generalize across model architectures
(GPT-2 and Qwen2.5) and task formats (question answering, fact verification,
and multi-hop reasoning).

The mechanistic account is not merely descriptive. We have shown that it can
be operationalized: TRACE, our mechanism-grounded auditing and intervention
framework, extracts internal evidence-to-answer traces, diagnoses specific
mechanism weaknesses, and applies matched interventions. On two independently
tested >1B-parameter models, TRACE reduces black-box reasoning failures by
57–75% without increasing false positives on safe samples.

Three findings define the current boundary of this approach. First, conflict
non-disclosure and evidence-gap unsupported answers are nearly eliminated —
the model learns to abstain or disclose when the evidence is insufficient or
contradictory. Second, misleading-driven errors are partially resistant,
consistent with the finding that QK routing to misleading evidence can
survive prompt-level intervention. Third, internal mechanism traces provide
more selective error detection than confidence scores, which fire
indiscriminately on most samples in smaller models.

The broader implication is that Transformer reasoning can be understood not
only through external behavior — what the model says, how confident it
appears — but through the internal computational pathway that produces
that behavior. This pathway is measurable, decomposable, causally
validatable, and actionable. We do not claim to have fully reverse-engineered
Transformer reasoning; the feature-level detail within each mechanism stage
remains for future work. What we have established is a structural skeleton
for evidence-to-answer computation, and the demonstration that understanding
this skeleton enables practical reduction of black-box failures — moving from
explaining Transformers toward making them more reliable.
