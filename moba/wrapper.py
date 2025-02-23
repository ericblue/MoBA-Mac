import torch
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from .config import MoBAConfig


def hf_to_fa(x: torch.Tensor):
    """
    Convert Hugging Face tensor format to Flash Attention format.

    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def fa_to_hf(x: torch.Tensor, batch: int):
    """
    Convert Flash Attention format back to Hugging Face format.

    Args:
        x (torch.Tensor): [batch * seqlen, heads, head_dim]

    Returns:
        torch.Tensor: [batch, heads, seqlen, head_dim]
    """
    return x.view(batch, -1, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)


def pytorch_attention(query, key, value, dropout=0.0, scaling=None):
    """
    Fallback PyTorch attention implementation (for systems without flash-attn).

    Args:
        query (torch.Tensor): [batch, q_len, heads, head_dim]
        key (torch.Tensor): [batch, kv_len, heads, head_dim]
        value (torch.Tensor): [batch, kv_len, heads, head_dim]
        dropout (float, optional): Dropout probability.
        scaling (float, optional): Scaling factor.

    Returns:
        torch.Tensor: Attention output [batch, q_len, heads, head_dim]
    """
    scaling = scaling if scaling else 1.0 / (query.shape[-1] ** 0.5)

    # Compute scaled dot-product attention
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Apply dropout if needed
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def moba_layer(
        moba_impl: Callable,
        moba_config: MoBAConfig,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    MoBA layer wrapper function.

    Args:
        query (torch.Tensor): [batch, q_heads, q_len, head_dim]
        key (torch.Tensor): [batch, kv_heads, kv_len, head_dim]
        value (torch.Tensor): [batch, kv_heads, kv_len, head_dim]
        dropout (float, optional): Dropout rate.
        scaling (float, optional): Scaling factor.

    Returns:
        Tuple: (attn_output, None)
    """
    assert module.is_causal
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape

    if q_len == kv_len:
        # Prefill phase
        query = hf_to_fa(query)
        key = hf_to_fa(key)
        value = hf_to_fa(value)

        # Adjust key/value dimensions
        kv_replicas = q_heads // kv_heads
        key = torch.repeat_interleave(key, kv_replicas, dim=1)
        value = torch.repeat_interleave(value, kv_replicas, dim=1)

        # Cumulative sequence length for MoBA
        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [kv_len] * batch, device=query.device),
            dim=0,
            dtype=torch.int32,
        )

        # Call MoBA implementation
        out = moba_impl(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=kv_len,
            moba_chunk_size=moba_config.moba_chunk_size,
            moba_topk=moba_config.moba_topk,
        )
    else:
        # Decode phase (Fallback to PyTorch Attention)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = pytorch_attention(query, key, value, dropout, scaling)

    return out, None
