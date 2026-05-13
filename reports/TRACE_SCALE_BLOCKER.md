# TRACE-Scale: Blocker Assessment

*May 2026*

---

## Status: Infrastructure Ready, >1B Models Not Available

The TRACE-Scale error reduction experiment is fully implemented at
`experiments/run_trace_scale.py`. The pipeline supports:
- 5-way comparison (raw, CoT, confidence, attention entropy, TRACE)
- Per-error-type outcome measurement
- Multiple model architectures

## Available Models (Fully Cached)

| Model | Params | Architecture |
|-------|--------|-------------|
| gpt2 | 124M | GPT-2 |
| gpt2-medium | 355M | GPT-2 |
| distilgpt2 | 82M | GPT-2 |
| pythia-70m | 70M | NeoX (fp16 NaN) |
| pythia-160m | 160M | NeoX (fp16 NaN) |
| Qwen2.5-0.5B | 494M | LLaMA |

## Unavailable Models (Download Blocked)

| Model | Params | Needed For |
|-------|--------|-----------|
| Qwen2.5-1.5B | 1.5B | Primary >1B validation |
| TinyLlama-1.1B | 1.1B | Cross-architecture >1B |
| Qwen2.5-3B | 3B | Strong output fidelity |

Blocked by: HuggingFace Hub SSL EOF errors (network infrastructure issue).

## What Was Tested at Available Scale

Qwen2.5-0.5B (494M) — the strongest fully-cached model:
- Error rate: 92.5% → 87.5% (5.4% reduction)
- Conservative prompting: 0% → 7.5% safe output rate
- TRACE selectivity preserved (per-error-type intervention)

## Honest Assessment

> TRACE-Scale is methodologically ready but empirically blocked by model-access
> infrastructure. The current evidence supports mechanism-grounded selective
> auditing and error-type-aware intervention, while large-scale error reduction
> remains an explicitly pending validation target. A 0.5B pilot shows 5.4%
> error reduction but is not used as a final claim because output fidelity
> limits measurable intervention effects.

> The infrastructure is ready to test the scale-dependent error-reduction
> hypothesis once the models are available.

## What To Do When Models Are Available

```bash
# Primary validation
python experiments/run_trace_scale.py --model Qwen/Qwen2.5-1.5B --limit-per-type 30

# Cross-architecture
python experiments/run_trace_scale.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --limit-per-type 30
```

Expected runtime on MPS (Apple GPU): ~15-30 min per model with 120 samples.
