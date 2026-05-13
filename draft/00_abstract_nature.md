# Abstract — Nature Version

*Draft — May 2026*

---

Transformer reasoning is often attributed to attention, yet how Transformers
internally convert evidence into answers remains unknown. Here we show that
evidence-to-answer processing follows a staged internal pathway: QK attention
routes evidence to relevant token positions, MLP layers causally transform this
evidence into answer-relevant reasoning states, and residual streams store
distributed intermediate representations before logit-level answer production.
We verify this mechanism across three model architectures and five task formats,
finding that QK routing discriminates direct from misleading evidence by 7.6–19.3×,
MLP output ablation causes 1.4–3.0× larger logit changes than attention ablation in 54
of 54 tested samples, and residual streams encode reasoning states with high
separability (S_X = 0.50–0.99, all permutation test p < 0.0001). We convert this
mechanistic discovery into TRACE (Transformer Reasoning Auditing through Causal
Evidence), a framework that extracts internal evidence-to-answer traces, diagnoses
black-box risks, and applies mechanism-matched interventions. On two independently
tested >1B-parameter models, TRACE-guided intervention reduces total black-box
reasoning failures by 57–75% without increasing false positives on safe samples,
with near-complete elimination of conflict non-disclosure and evidence-gap
unsupported answers. Misleading-driven errors remain partially resistant,
consistent with the documented format-sensitivity of QK-based evidence routing.
These results establish that Transformer internal reasoning mechanisms can be
measured, decomposed, and causally utilized to reduce black-box failures —
moving beyond explaining model behavior toward mechanism-grounded reliability
improvement.
