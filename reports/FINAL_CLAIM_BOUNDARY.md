# Final Claim Boundary

*Frozen: May 2026*

---

## CAN CLAIM (High Confidence)

| # | Claim | Evidence |
|---|-------|----------|
| C1 | Evidence-to-answer follows a staged mechanism: QK→MLP→X_l→logits | Cross-model (3), cross-format (5), causal (ablation + patching) |
| C2 | QK attention routes evidence; 7.6–19.3× direct/misleading discrimination | 3/3 models, cross-architecture |
| C3 | QK reconstruction verified: softmax(QK^T/√d) ≈ A_model | All 12 GPT-2 layers, tol 10^{-5} |
| C4 | MLP is the dominant causal pathway for evidence-to-answer transformation | 1.4–3.0× over attention, 51/51 cross-format |
| C5 | MLP activation patching recovers correct answers from corrupted evidence | 77.1% avg recovery |
| C6 | Residual streams encode reasoning states | S_X=0.50–0.99, all p<0.0001, all confounds ruled out |
| C7 | S_X is not a template artifact | Shuffled=20.4%, random=15.0%, token-count=41.0% |
| C8 | TRACE reduces black-box failures in controlled mechanism-matched interventions on >1B models | 57–75% reduction, 3 models, 2 architectures |
| C9 | TRACE sharply reduces conflict non-disclosure and evidence-gap unsupported answers | 95–97% reduction on both error types |
| C10 | TRACE does not increase direct-evidence false positives | No increase observed (2/40 in both raw and TRACE) |
| C11 | TRACE effect is not prompt engineering | Mismatched 1.9× worse (p=0.0013), random 1.5× worse (p=0.0212) |
| C12 | TRACE V3.1 operates without gold labels | Uses R_QK + confidence + S_X probe |
| C13 | Misleading evidence hijacks QK routing (13–108× cue dominance) | Dual-span R_QK analysis |
| C14 | Mechanism shifts from routing-dominant to state-encoding-dominant with depth | α↓ (0.306→0.197), γ↑ (0.249→0.445) |
| C15 | Visible CoT amplifies but does not replace internal mechanism | CoT +0.025 S_X, misleading +0.000 |

## CAN DISCUSS (Medium Confidence)

| # | Claim | Qualification |
|---|-------|---------------|
| D1 | Mechanism generalizes across all Transformer architectures | Tested GPT-2 + Qwen2.5 + LLaMA; broader validation pending |
| D2 | MLP two-phase computation (early/late suppression, middle construction) | Observed GPT-2 12L; cross-model validation pending |
| D3 | TRACE V3.1 is the current best autonomous trigger | 71% fire rate still above deployment target |
| D4 | Autonomous trigger calibration has a safety–utility frontier | V3.2: fire↓→error↑; stronger signals needed |
| D5 | LLaMA base failure reflects instruction-following requirement | Base model ignores corrective prompts; Instruct model responds |
| D6 | Cross-format validation supports mechanism robustness | FEVER/HotpotQA are format adaptations of controlled samples, not benchmark evaluations |

## MUST NOT CLAIM (Insufficient Evidence)

| # | Overclaim | Why |
|---|-----------|-----|
| X1 | Full reverse-engineering of Transformer reasoning | Only structural skeleton identified |
| X2 | TRACE eliminates all black-box failures | Misleading remains partially resistant |
| X3 | TRACE works universally across all models and tasks | 5 architectures, 5 formats, controlled tasks |
| X4 | V3.1 is deployment-ready autonomous TRACE | 71% fire rate, calibration frontier not resolved |
| X5 | Two-phase MLP is a universal Transformer property | Low confidence, single-model observation |
| X6 | TRACE replaces need for external evaluation | Complementary tool |
| X7 | Zero false positives is a universal guarantee | n=40, Wilson upper bound 8.8% |
| X8 | Current FEVER/HotpotQA results are benchmark evaluations | Format adaptations, not official benchmarks |
