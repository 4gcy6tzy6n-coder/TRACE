"""C_do: Causal intervention scores via token ablation and attention masking.

v0.2: Adds AttentionMaskHook for true attention-level masking (not embedding-zeroing).
Also adds true_attention_ablation using the hook-based approach.
"""

import torch
import torch.nn.functional as F
from src.model_loader import ActivationCache, run_forward, get_token_logit


def evidence_token_ablation(
    model,
    tokenizer,
    cache: ActivationCache,
    prompt: str,
    evidence_span: str,
    gold_answer: str,
    device: str = "cpu",
) -> dict:
    """Remove gold evidence span from prompt and measure logit drop."""
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = get_token_logit(orig_result["logits"], gold_token_id)

    ablated_prompt = prompt.replace(evidence_span, "")
    ablated_prompt = " ".join(ablated_prompt.split())

    abl_result = run_forward(model, tokenizer, ablated_prompt, cache, device)
    abl_logit = get_token_logit(abl_result["logits"], gold_token_id)

    logit_drop = orig_logit - abl_logit

    return {
        "original_logit": orig_logit,
        "ablated_logit": abl_logit,
        "logit_drop": logit_drop,
        "C_token_ablation": _normalize_logit_drop(logit_drop),
    }


def attention_evidence_ablation(
    model,
    tokenizer,
    cache: ActivationCache,
    prompt: str,
    evidence_positions: list[int],
    answer_positions: list[int],
    gold_answer: str,
    device: str = "cpu",
) -> dict:
    """Mask attention from answer to evidence positions (v0.1 embedding-zeroing method).

    Kept for baseline comparison. Prefer true_attention_ablation for v0.2.
    """
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = get_token_logit(orig_result["logits"], gold_token_id)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    embed_layer = model.transformer.wte
    embeddings = embed_layer(input_ids)

    for pos in evidence_positions:
        if pos < embeddings.shape[1]:
            embeddings[0, pos] = 0.0

    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings)

    masked_logit = outputs.logits[0, -1, gold_token_id].item()

    logit_drop = orig_logit - masked_logit

    return {
        "original_logit": orig_logit,
        "masked_logit": masked_logit,
        "logit_drop": logit_drop,
        "C_attention_ablation": _normalize_logit_drop(logit_drop),
    }


class AttentionMaskHook:
    """Context manager that overrides attention to block evidence→answer routing.

    Intercepts each attention layer's forward pass, recomputes Q/K/V from
    the module's own weights, and replaces the attention with a version
    where answer→evidence attention scores are set to -inf.

    When evidence_positions and answer_positions are both empty, this
    acts as a no-op identity check (should produce identical logits).

    Usage:
        with AttentionMaskHook(model, evidence_pos, answer_pos, device):
            result = run_forward(model, tokenizer, prompt, cache, device)
    """

    def __init__(
        self,
        model,
        evidence_positions: list[int],
        answer_positions: list[int],
        device: str = "cpu",
    ):
        self.model = model
        self.evidence_positions = evidence_positions
        self.answer_positions = answer_positions
        self.device = device
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_layer_hook(self, layer_idx: int):
        block = self.model.transformer.h[layer_idx]
        attn = block.attn
        c_attn_w = attn.c_attn.weight.detach()   # [hidden, 3*hidden] Conv1D layout
        c_attn_b = attn.c_attn.bias.detach()      # [3*hidden]
        c_proj_w = attn.c_proj.weight.detach()    # [hidden, hidden]
        c_proj_b = attn.c_proj.bias.detach()      # [hidden]
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        evidence_set = set(self.evidence_positions)
        answer_set = set(self.answer_positions)
        device = self.device

        def hook(module, input, output):
            hidden_states = input[0]  # [batch, seq, hidden_dim]
            batch, seq_len, hidden_dim = hidden_states.shape

            # Compute QKV: x @ weight + bias (Conv1D forward)
            # weight is [hidden_dim, 3*hidden_dim]
            qkv = hidden_states @ c_attn_w + c_attn_b           # [batch, seq, 3*hidden]
            q, k, v = qkv.split(hidden_dim, dim=-1)              # each [batch, seq, hidden]

            # Reshape to multi-head: [batch, heads, seq, head_dim]
            q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

            # Attention scores
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask (lower triangular)
            causal = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            causal = causal.view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

            # Evidence mask: block attention from answer positions to evidence positions
            if evidence_set and answer_set:
                for p in answer_set:
                    for j in evidence_set:
                        if 0 <= p < seq_len and 0 <= j < seq_len:
                            attn_scores[:, :, p, j] = float("-inf")

            attn_weights = torch.softmax(attn_scores, dim=-1)

            # A @ V
            attn_out = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(
                batch, seq_len, hidden_dim
            )  # [batch, seq, hidden]

            # Output projection: x @ c_proj_w + c_proj_b
            attn_out = attn_out @ c_proj_w + c_proj_b

            return (attn_out, attn_weights)

        return hook

    def __enter__(self):
        for layer_idx in range(len(self.model.transformer.h)):
            attn_module = self.model.transformer.h[layer_idx].attn
            handle = attn_module.register_forward_hook(
                self._make_layer_hook(layer_idx)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def true_attention_ablation(
    model,
    tokenizer,
    cache: ActivationCache,
    prompt: str,
    evidence_positions: list[int],
    answer_positions: list[int],
    gold_answer: str,
    device: str = "cpu",
) -> dict:
    """Run forward with true attention masking via AttentionMaskHook.

    Blocks attention from answer positions to evidence positions across
    all layers and heads. Measures logit drop vs original forward.

    Also validates that the hook with empty positions produces identical
    logits to the original forward (sanity check).

    Returns:
        dict with original_logit, masked_logit, logit_drop, C_attention_ablation.
    """
    gold_token_id = tokenizer.encode(gold_answer, add_special_tokens=False)[0]

    # Original run (no mask)
    orig_result = run_forward(model, tokenizer, prompt, cache, device)
    orig_logit = get_token_logit(orig_result["logits"], gold_token_id)

    # Identity check: hook with empty positions should match original
    with AttentionMaskHook(model, [], [], device):
        id_result = run_forward(model, tokenizer, prompt, cache, device)
    id_logit = get_token_logit(id_result["logits"], gold_token_id)
    identity_check_passed = abs(orig_logit - id_logit) < 0.01

    # True ablation: mask answer→evidence attention
    with AttentionMaskHook(model, evidence_positions, answer_positions, device):
        masked_result = run_forward(model, tokenizer, prompt, cache, device)
    masked_logit = get_token_logit(masked_result["logits"], gold_token_id)

    logit_drop = orig_logit - masked_logit

    return {
        "original_logit": orig_logit,
        "masked_logit": masked_logit,
        "logit_drop": logit_drop,
        "C_attention_ablation": _normalize_logit_drop(logit_drop),
        "identity_check_passed": identity_check_passed,
    }


def compute_c_do(
    model,
    tokenizer,
    cache: ActivationCache,
    prompt: str,
    evidence_span: str,
    evidence_positions: list[int],
    answer_positions: list[int],
    gold_answer: str,
    device: str = "cpu",
    use_true_attn_ablation: bool = True,
) -> dict:
    """Compute C_do combining token ablation and (true) attention ablation.

    Args:
        use_true_attn_ablation: If True, use AttentionMaskHook (v0.2).
            If False, use embedding-zeroing (v0.1 fallback).

    Returns:
        dict with C_token_ablation, C_attention_ablation, C_do.
    """
    token_result = evidence_token_ablation(
        model, tokenizer, cache, prompt, evidence_span, gold_answer, device
    )

    if use_true_attn_ablation:
        attn_result = true_attention_ablation(
            model, tokenizer, cache, prompt,
            evidence_positions, answer_positions, gold_answer, device,
        )
    else:
        attn_result = attention_evidence_ablation(
            model, tokenizer, cache, prompt,
            evidence_positions, answer_positions, gold_answer, device,
        )

    c_do = (token_result["C_token_ablation"] + attn_result["C_attention_ablation"]) / 2.0

    return {
        "C_token_ablation": token_result["C_token_ablation"],
        "C_attention_ablation": attn_result["C_attention_ablation"],
        "C_do": c_do,
        "token_ablation_detail": token_result,
        "attention_ablation_detail": attn_result,
    }


def _normalize_logit_drop(logit_drop: float, scale: float = 5.0) -> float:
    """Normalize logit drop to [0, 1] using sigmoid-like scaling.

    A drop of 0 → 0.0, drop of `scale` → ~0.73, large drop → ~1.0.
    Negative drops (logit increased after ablation) → 0.0.
    """
    if logit_drop <= 0:
        return 0.0
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-logit_drop / scale))) - 1.0)
