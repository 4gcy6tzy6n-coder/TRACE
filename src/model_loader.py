"""Load GPT-2 with hooks to extract internal activations including Q/K/V.

v0.2: Adds c_attn hook for raw Q/K/V extraction, split/reshape utilities,
and verification that recomputed attention matches model attention weights.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class ActivationCache:
    """Collects activations via forward hooks during a model forward pass.

    v0.2 additions:
        - c_attn_outputs: raw fused QKV per layer [batch, seq, 3*hidden]
        - q_per_layer, k_per_layer, v_per_layer: split per-head [B, heads, seq, head_dim]
    """

    def __init__(self):
        self.attentions: list[torch.Tensor] = []
        self.hidden_states: list[torch.Tensor] = []
        self.mlp_outputs: list[torch.Tensor] = []
        self.logits: Optional[torch.Tensor] = None
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        # v0.2: Q/K/V extraction
        self.c_attn_outputs: list[torch.Tensor] = []
        self.q_per_layer: list[torch.Tensor] = []
        self.k_per_layer: list[torch.Tensor] = []
        self.v_per_layer: list[torch.Tensor] = []

    def _attention_hook(self, module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                self.attentions.append(attn_weights.detach())

    def _hidden_hook(self, module, input, output):
        if isinstance(output, tuple):
            self.hidden_states.append(output[0].detach())
        else:
            self.hidden_states.append(output.detach())

    def _mlp_hook(self, module, input, output):
        self.mlp_outputs.append(output.detach())

    def _c_attn_hook(self, module, input, output):
        self.c_attn_outputs.append(output.detach())

    def clear(self):
        self.attentions.clear()
        self.hidden_states.clear()
        self.mlp_outputs.clear()
        self.logits = None
        self.c_attn_outputs.clear()
        self.q_per_layer.clear()
        self.k_per_layer.clear()
        self.v_per_layer.clear()


def load_model(model_name: str = "gpt2", device: str = "cpu"):
    """Load tokenizer and GPT-2 model with comprehensive activation hooks.

    Hooks registered per block:
        - block.attn.c_attn: raw fused QKV output
        - block.attn: attention weights
        - block: post-block hidden state
        - block.mlp: MLP activations

    Args:
        model_name: HuggingFace model ID (default: 'gpt2').
        device: 'cpu', 'cuda', or 'mps'.

    Returns:
        (model, tokenizer, cache) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    cache = ActivationCache()

    for layer_idx, block in enumerate(model.transformer.h):
        # v0.2: c_attn hook for raw Q/K/V
        handle_c_attn = block.attn.c_attn.register_forward_hook(cache._c_attn_hook)
        cache.hooks.append(handle_c_attn)

        # Attention weights hook
        handle_attn = block.attn.register_forward_hook(cache._attention_hook)
        cache.hooks.append(handle_attn)

        # Hidden state hook (after the block)
        handle_hidden = block.register_forward_hook(cache._hidden_hook)
        cache.hooks.append(handle_hidden)

        # MLP hook
        handle_mlp = block.mlp.register_forward_hook(cache._mlp_hook)
        cache.hooks.append(handle_mlp)

    return model, tokenizer, cache


def split_qkv(
    c_attn_output: torch.Tensor,
    num_heads: int = 12,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused c_attn output into Q, K, V and reshape to multi-head.

    GPT-2's c_attn is Conv1D(hidden, 3*hidden). Output is [batch, seq, 3*hidden]
    with Q, K, V concatenated along the last dimension.

    Args:
        c_attn_output: [batch, seq, 3*hidden_dim] from c_attn forward.
        num_heads: Number of attention heads (12 for gpt2).
        head_dim: Dimension per head (64 for gpt2, hidden=768).

    Returns:
        q, k, v: each [batch, heads, seq, head_dim].
    """
    hidden_dim = num_heads * head_dim
    q, k, v = c_attn_output.split(hidden_dim, dim=-1)

    q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
    k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(v.shape[0], v.shape[1], num_heads, head_dim).transpose(1, 2)

    return q, k, v


def process_qkv_cache(
    cache: ActivationCache,
    num_heads: int = 12,
    head_dim: int = 64,
) -> None:
    """Populate cache.q/k/v_per_layer from raw c_attn_outputs."""
    cache.q_per_layer.clear()
    cache.k_per_layer.clear()
    cache.v_per_layer.clear()
    for c_attn_out in cache.c_attn_outputs:
        q, k, v = split_qkv(c_attn_out, num_heads, head_dim)
        cache.q_per_layer.append(q)
        cache.k_per_layer.append(k)
        cache.v_per_layer.append(v)


def verify_qk_reconstruction(
    q: torch.Tensor,
    k: torch.Tensor,
    model_attn_weights: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Verify that softmax(QK^T / sqrt(d)) matches model's attention weights.

    Applies causal masking to match GPT-2's autoregressive attention.

    Args:
        q: [batch, heads, seq, head_dim].
        k: [batch, heads, seq, head_dim].
        model_attn_weights: [batch, heads, seq, seq] from model output.
        atol: Absolute tolerance for allclose.

    Returns:
        True if recomputed attention matches model output within tolerance.
    """
    head_dim = q.shape[-1]
    seq_len = q.shape[2]
    scale = head_dim ** -0.5

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal (lower-triangular) mask
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
    ).view(1, 1, seq_len, seq_len)
    attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    computed_attn = torch.softmax(attn_scores, dim=-1)

    return bool(torch.allclose(computed_attn, model_attn_weights, atol=atol))


def run_forward(
    model,
    tokenizer,
    prompt: str,
    cache: ActivationCache,
    device: str = "cpu",
    process_qkv: bool = True,
) -> dict:
    """Run forward pass and capture activations including Q/K/V.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        prompt: Input text.
        cache: ActivationCache instance.
        device: Device string.
        process_qkv: If True, split c_attn into per-head Q/K/V after forward.

    Returns:
        dict with input_ids, tokens, logits, attentions, hidden_states,
        mlp_outputs, q_per_layer, k_per_layer, v_per_layer.
    """
    cache.clear()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    cache.logits = outputs.logits.detach()

    if process_qkv:
        process_qkv_cache(cache)

    tokens = [tokenizer.decode(tok) for tok in input_ids[0]]

    return {
        "input_ids": input_ids,
        "tokens": tokens,
        "logits": outputs.logits.detach(),
        "attentions": cache.attentions,
        "hidden_states": cache.hidden_states,
        "mlp_outputs": cache.mlp_outputs,
        "q_per_layer": cache.q_per_layer,
        "k_per_layer": cache.k_per_layer,
        "v_per_layer": cache.v_per_layer,
    }


def get_token_logit(
    logits: torch.Tensor,
    token_id: int,
    position: int = -1,
) -> float:
    """Get the logit value for a specific token at a given position."""
    return logits[0, position, token_id].item()


def remove_hooks(cache: ActivationCache) -> None:
    for hook in cache.hooks:
        hook.remove()
    cache.hooks.clear()
