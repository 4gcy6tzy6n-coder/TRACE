# Internal CoT Index (ICI)

Quantify whether a Transformer model has an internal chain-of-thought trajectory by measuring four signals from internal activations.

## Formula

```
ICI = alpha * R_QK + beta * M_AV + gamma * S_X + delta * C_do
```

| Score | Meaning | Transformer Signal |
|-------|---------|-------------------|
| R_QK | Does attention route to correct evidence? | Q, K, attention weights |
| M_AV | Does evidence value drive the answer? | Attention, V, W_O, logits |
| S_X | Does residual stream encode reasoning state? | Hidden states, MLP |
| C_do | Does intervening on internal paths change the answer? | Ablation / logit change |

## Project Structure

```
internal_cot_index/
├── data/
│   └── toy_reasoning.jsonl       # 50 controlled samples (5 types)
├── src/
│   ├── model_loader.py           # GPT-2 loading with activation hooks
│   ├── token_mapper.py           # Evidence/answer span → token indices
│   ├── qk_routing_score.py       # R_QK: attention routing score
│   ├── av_message_score.py       # M_AV: evidence-to-answer score
│   ├── residual_state_score.py   # S_X: linear probe on hidden states
│   ├── causal_intervention.py    # C_do: ablation-based causal score
│   ├── ici_calculator.py         # ICI aggregation and output
│   └── utils.py                  # Shared utilities
├── experiments/
│   ├── run_collect_activations.py # Collect and save activations
│   ├── run_ici_eval.py            # Full ICI evaluation pipeline
│   ├── run_ablation.py            # Ablation experiments
│   └── run_probe_training.py      # Train and evaluate linear probes
└── reports/
    └── ici_results.md             # Results table
```

## Quick Start

```bash
# Install dependencies
pip install torch transformers scikit-learn numpy

# Run full ICI evaluation (50 samples)
python experiments/run_ici_eval.py

# Quick test on 5 samples (skip slow C_do)
python experiments/run_ici_eval.py --limit 5 --skip-cdo

# Run only ablation experiments
python experiments/run_ablation.py --limit 10

# Train probe and compute S_X
python experiments/run_probe_training.py

# Collect and save activations
python experiments/run_collect_activations.py
```

## Sample Format

```json
{
  "id": "case_001",
  "evidence": ["Doc A: ...", "Doc B: ..."],
  "question": "When was the model released?",
  "gold_answer": "2023",
  "gold_evidence_span": "released in 2023",
  "reasoning_type": "direct_evidence",
  "gold_thought_steps": ["...", "..."],
  "label": "faithful"
}
```

## Reasoning Types

| Type | Description | Count |
|------|-------------|-------|
| direct_evidence | Evidence directly supports answer | 10 |
| conflict | Multiple documents conflict | 10 |
| evidence_gap | Evidence insufficient to answer | 10 |
| misleading_hint | Misleading cues present | 10 |
| multi_step | Requires multi-step reasoning | 10 |

## v0.2 Features (Current)

| Component | Description |
|-----------|-------------|
| Q/K/V Extraction | Per-head Q/K/V from fused c_attn, verified via attention reconstruction |
| True M_AV | A×V×W_O×W_U circuit — evidence V vectors → answer logit contribution |
| AttentionMaskHook | True attention-level masking (not embedding-zeroing), identity-verified |
| Layer/Head Analysis | Per-head R_QK, per-head M_AV, layer-wise S_X probes, heatmap export |
| Activation Patching | Residual stream patching clean→corrupted, layer sweep, recovery measurement |
| CoT Comparison | ICI with vs without chain-of-thought prompt, per-type analysis |

### Quick Start (v0.2)

```bash
# Full v0.2 evaluation with true M_AV, per-head analysis, layer-wise S_X
python experiments/run_ici_eval.py --v2 --heatmap --layerwise-sx --skip-cdo

# CoT vs no-CoT comparison
python experiments/run_cot_comparison.py --limit 20

# Activation patching
python experiments/run_patching.py --limit 3

# True attention ablation
python experiments/run_ablation.py --limit 10
```

### Key v0.2 Findings

- **Q/K/V verified**: All 12 layers pass `softmax(QK^T/√d) ≈ A_model`
- **True M_AV separates from proxy**: Captures negative V contributions proxy can't see
- **Evidence routing heads**: Concentrated in Layer 0 (heads 1,3,4,5)
- **Patching shows causality**: Single-layer residual patch can fully recover correct answer
- **CoT boosts ICI**: +0.016 direct_evidence, +0.018 conflict; S_X rises from 0.25→0.31

## v0.1 Features (Baseline)

- Model: GPT-2 (124M)
- Dataset: 50 controlled samples
- Extracts: attention weights, hidden states, logits
- Computes: R_QK, simplified M_AV, S_X (probe), simplified C_do
- Output: ICI score per sample with correlation analysis
