"""Architecture-aware model loader for multi-model ICI evaluation.

Supports: GPT-2 family, Pythia, Qwen2.5, LLaMA 3.
Detects model architecture and registers appropriate hooks for Q/K/V extraction.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class ActivationCache:
    """Collects activations during forward pass. Architecture-agnostic."""

    def __init__(self):
        self.attentions: list[torch.Tensor] = []
        self.hidden_states: list[torch.Tensor] = []
        self.mlp_outputs: list[torch.Tensor] = []
        self.logits: Optional[torch.Tensor] = None
        self.q_per_layer: list[torch.Tensor] = []
        self.k_per_layer: list[torch.Tensor] = []
        self.v_per_layer: list[torch.Tensor] = []
        self.c_attn_outputs: list[torch.Tensor] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Architecture metadata
        self.arch_type: str = "unknown"
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        self.hidden_dim: int = 0

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

    def _fused_qkv_hook(self, module, input, output):
        """For GPT-2/Pythia: fused c_attn/query_key_value output."""
        self.c_attn_outputs.append(output.detach())

    def _q_hook(self, module, input, output):
        self.q_per_layer.append(output.detach())

    def _k_hook(self, module, input, output):
        self.k_per_layer.append(output.detach())

    def _v_hook(self, module, input, output):
        self.v_per_layer.append(output.detach())

    def clear(self):
        self.attentions.clear()
        self.hidden_states.clear()
        self.mlp_outputs.clear()
        self.logits = None
        self.q_per_layer.clear()
        self.k_per_layer.clear()
        self.v_per_layer.clear()
        self.c_attn_outputs.clear()


def detect_architecture(model) -> dict:
    """Detect model architecture and return relevant attributes."""
    config = model.config
    arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "unknown"

    info = {"arch": arch}

    # Try to extract common parameters
    if hasattr(config, 'n_embd'):           # GPT-2
        info["hidden_dim"] = config.n_embd
        info["num_heads"] = config.n_head
        info["num_layers"] = config.n_layer
        info["family"] = "gpt2"
    elif hasattr(config, 'hidden_size'):
        info["hidden_dim"] = config.hidden_size
        if hasattr(config, 'num_attention_heads'):
            info["num_heads"] = config.num_attention_heads
        elif hasattr(config, 'n_head'):
            info["num_heads"] = config.n_head
        if hasattr(config, 'num_hidden_layers'):
            info["num_layers"] = config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            info["num_layers"] = config.n_layer
        if "Llama" in arch or "Qwen" in arch:
            info["family"] = "llama"
        elif "GPTNeoX" in arch or "NeoX" in arch:
            info["family"] = "neox"
        elif "Pythia" in arch:
            info["family"] = "neox"
        else:
            info["family"] = "unknown"
    elif hasattr(config, 'd_model'):        # T5-style
        info["hidden_dim"] = config.d_model
        info["num_heads"] = config.num_heads
        info["num_layers"] = config.num_layers
        info["family"] = "t5"

    if "num_heads" in info and "hidden_dim" in info:
        info["head_dim"] = info["hidden_dim"] // info["num_heads"]
    else:
        info["head_dim"] = 64  # default

    return info


def load_model_v04(model_name: str = "gpt2", device: str = "cpu"):
    """Load any HF model with architecture-aware hooks.

    Args:
        model_name: HuggingFace model ID (gpt2, gpt2-medium, pythia-160m, Qwen/Qwen2.5-0.5B, etc.).
        device: 'cpu', 'cuda', or 'mps'.

    Returns:
        (model, tokenizer, cache, arch_info) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    arch_info = detect_architecture(model)
    cache = ActivationCache()
    cache.arch_type = arch_info["family"]
    cache.num_heads = arch_info.get("num_heads", 12)
    cache.head_dim = arch_info.get("head_dim", 64)
    cache.hidden_dim = arch_info.get("hidden_dim", 768)

    family = arch_info["family"]
    print(f"Architecture: {arch_info['arch']} (family={family}, layers={arch_info.get('num_layers')}, heads={arch_info.get('num_heads')})")

    # Find the transformer layers
    layers = _get_transformer_layers(model, family)
    cache.num_layers = len(layers)

    for layer_idx, layer in enumerate(layers):
        # Register hooks based on architecture
        if family == "gpt2":
            _register_gpt2_hooks(layer, cache)
        elif family == "neox":
            _register_neox_hooks(layer, cache)
        elif family == "llama" or "qwen" in model_name.lower():
            _register_llama_hooks(layer, cache)
        else:
            _register_fallback_hooks(layer, cache, family)

    return model, tokenizer, cache, arch_info


def _get_transformer_layers(model, family: str):
    """Get the list of transformer layers based on architecture family."""
    if family == "gpt2":
        return model.transformer.h
    if family == "neox":
        if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            return model.gpt_neox.layers
    if family == "llama":
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
    # Generic fallbacks
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    # Generic search
    for attr in ['transformer', 'model', 'gpt_neox']:
        if hasattr(model, attr):
            base = getattr(model, attr)
            for sub_attr in ['h', 'layers', 'blocks']:
                if hasattr(base, sub_attr):
                    return getattr(base, sub_attr)
    raise ValueError(f"Cannot find transformer layers for family={family}")


def _register_gpt2_hooks(block, cache):
    """GPT-2: fused c_attn, separate attn/mlp."""
    handle_c_attn = block.attn.c_attn.register_forward_hook(cache._fused_qkv_hook)
    cache.hooks.append(handle_c_attn)
    handle_attn = block.attn.register_forward_hook(cache._attention_hook)
    cache.hooks.append(handle_attn)
    handle_hidden = block.register_forward_hook(cache._hidden_hook)
    cache.hooks.append(handle_hidden)
    handle_mlp = block.mlp.register_forward_hook(cache._mlp_hook)
    cache.hooks.append(handle_mlp)


def _register_neox_hooks(layer, cache):
    """Pythia/GPT-NeoX: fused query_key_value in attention."""
    attn = layer.attention
    if hasattr(attn, 'query_key_value'):
        handle_qkv = attn.query_key_value.register_forward_hook(cache._fused_qkv_hook)
        cache.hooks.append(handle_qkv)
    handle_attn = attn.register_forward_hook(cache._attention_hook)
    cache.hooks.append(handle_attn)
    handle_hidden = layer.register_forward_hook(cache._hidden_hook)
    cache.hooks.append(handle_hidden)
    if hasattr(layer, 'mlp'):
        handle_mlp = layer.mlp.register_forward_hook(cache._mlp_hook)
        cache.hooks.append(handle_mlp)


def _register_llama_hooks(layer, cache):
    """LLaMA/Qwen: separate q_proj, k_proj, v_proj."""
    attn = layer.self_attn
    # Try separate Q/K/V projections
    if hasattr(attn, 'q_proj'):
        handle_q = attn.q_proj.register_forward_hook(cache._q_hook)
        cache.hooks.append(handle_q)
    if hasattr(attn, 'k_proj'):
        handle_k = attn.k_proj.register_forward_hook(cache._k_hook)
        cache.hooks.append(handle_k)
    if hasattr(attn, 'v_proj'):
        handle_v = attn.v_proj.register_forward_hook(cache._v_hook)
        cache.hooks.append(handle_v)
    # Fallback: fused projection (Pythia/Qwen2)
    if hasattr(attn, 'qkv_proj'):
        handle_qkv = attn.qkv_proj.register_forward_hook(cache._fused_qkv_hook)
        cache.hooks.append(handle_qkv)
    elif hasattr(attn, 'query_key_value'):
        handle_qkv = attn.query_key_value.register_forward_hook(cache._fused_qkv_hook)
        cache.hooks.append(handle_qkv)

    # Attention output
    handle_attn = attn.register_forward_hook(cache._attention_hook)
    cache.hooks.append(handle_attn)
    # Hidden state
    handle_hidden = layer.register_forward_hook(cache._hidden_hook)
    cache.hooks.append(handle_hidden)
    # MLP
    if hasattr(layer, 'mlp'):
        handle_mlp = layer.mlp.register_forward_hook(cache._mlp_hook)
        cache.hooks.append(handle_mlp)


def _register_fallback_hooks(block, cache, family):
    """Generic fallback hook registration."""
    # Try to find attention
    for attn_name in ['attn', 'attention', 'self_attn', 'self_attention']:
        if hasattr(block, attn_name):
            attn = getattr(block, attn_name)
            handle_attn = attn.register_forward_hook(cache._attention_hook)
            cache.hooks.append(handle_attn)
            # Try QKV
            for qkv_name in ['c_attn', 'query_key_value', 'qkv_proj']:
                if hasattr(attn, qkv_name):
                    handle_qkv = getattr(attn, qkv_name).register_forward_hook(cache._fused_qkv_hook)
                    cache.hooks.append(handle_qkv)
                    break
            break

    handle_hidden = block.register_forward_hook(cache._hidden_hook)
    cache.hooks.append(handle_hidden)

    for mlp_name in ['mlp', 'feed_forward', 'ffn']:
        if hasattr(block, mlp_name):
            handle_mlp = getattr(block, mlp_name).register_forward_hook(cache._mlp_hook)
            cache.hooks.append(handle_mlp)
            break


def split_qkv(
    c_attn_output: torch.Tensor,
    num_heads: int = 12,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV into per-head Q, K, V."""
    hidden_dim = num_heads * head_dim
    q, k, v = c_attn_output.split(hidden_dim, dim=-1)
    q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
    k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(v.shape[0], v.shape[1], num_heads, head_dim).transpose(1, 2)
    return q, k, v


def process_qkv_cache(cache: ActivationCache) -> None:
    """Populate q/k/v_per_layer from c_attn or separate Q/K/V hooks."""
    cache.q_per_layer.clear()
    cache.k_per_layer.clear()
    cache.v_per_layer.clear()

    if cache.c_attn_outputs:
        # Fused QKV: split
        for c_attn_out in cache.c_attn_outputs:
            q, k, v = split_qkv(c_attn_out, cache.num_heads, cache.head_dim)
            cache.q_per_layer.append(q)
            cache.k_per_layer.append(k)
            cache.v_per_layer.append(v)
    elif cache.q_per_layer and cache.k_per_layer and cache.v_per_layer:
        # Already split (LLaMA-style): reshape to multi-head
        num_heads = cache.num_heads
        head_dim = cache.head_dim
        # Reshape collected Q/K/V
        q_list, k_list, v_list = [], [], []
        for i in range(len(cache.q_per_layer)):
            q = cache.q_per_layer[i]
            q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
            q_list.append(q)
            k = cache.k_per_layer[i]
            k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
            k_list.append(k)
            v = cache.v_per_layer[i]
            v = v.view(v.shape[0], v.shape[1], num_heads, head_dim).transpose(1, 2)
            v_list.append(v)
        cache.q_per_layer = q_list
        cache.k_per_layer = k_list
        cache.v_per_layer = v_list


def run_forward(
    model,
    tokenizer,
    prompt: str,
    cache: ActivationCache,
    device: str = "cpu",
    process_qkv: bool = True,
) -> dict:
    """Run forward pass and collect activations."""
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


def get_token_logit(logits, token_id, position=-1):
    return logits[0, position, token_id].item()


def remove_hooks(cache: ActivationCache) -> None:
    for hook in cache.hooks:
        hook.remove()
    cache.hooks.clear()
