# Nature Gaps: Closed vs Remaining

*May 2026*

---

## Closed (This Session)

| Gap | Evidence | Status |
|-----|----------|--------|
| G4: Intervention ablation | Full (34.7%) vs no-intv (97.5%) p<0.0001; vs mismatched (64.5%) p=0.0013; vs random (50.4%) p=0.0212 | **CLOSED** |
| G1: Irregular allocation protocol | Pre-specified n=(40,40,41,40); n=41 breaks identical-CI problem | **CLOSED** |
| Statistical rigor | Wilson CI, bootstrap CI, McNemar, raw counts shown | **CLOSED** |
| G5: Model-wise table | 1.5B complete; 3B directional consistency | **PARTIAL** |

## Remaining

| Gap | Blocker |
|-----|---------|
| G1: Non-Qwen >1B replication | Model access (LLaMA gated, Gemma gated, Qwen2.5-3B needs MPS) |
| G2: Large stratified samples (500-700/model) | Data generation capacity |
| G3: Stronger baselines (self-consistency, verifier, LLM-judge) | Run time + API access for LLM judge |
| G6: Misleading failure mechanism analysis | Run on Qwen2.5-1.5B (cached) |
| G7: MLP feature-level analysis | Run on GPT-2 (cached) |

## Intervention Ablation Result (New)

| Variant | Errors/121 | Rate [95% CI] | vs Full |
|---------|-----------|---------------|---------|
| TRACE_full | 42 | 34.7% [26.8, 43.5] | — |
| no_intervention | 118 | 97.5% [93.0, 99.2] | p<0.0001 *** |
| mismatched | 78 | 64.5% [55.6, 72.4] | p=0.0013 ** |
| random | 61 | 50.4% [41.6, 59.2] | p=0.0212 * |

Proves: TRACE is not prompt engineering (mismatched is 1.9× worse), not random
(random is 1.5× worse), and requires mechanism-matched intervention selection.
