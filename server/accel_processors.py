"""
I2I-Aware Acceleration Processors — server/accel_processors.py
==============================================================
Custom diffusers attention processors:

1. Self-attention (attn1): ALWAYS independent per candidate, using PyTorch 2.0
   F.scaled_dot_product_attention (Flash/Memory-Efficient Attention) to avoid
   materializing [batch, heads, 4096, 4096] attention matrices.
2. Cross-attention (attn2), step <= r: each candidate computes its OWN cross-attention
   with FP32 softmax over text dimension (77 tokens, ~80 MB) to eliminate FP16 NaN.
3. Cross-attention (attn2), step > r: temporal caching — cache attn2 output, refresh
   every 2 steps. Reuse cached pre-residual output with current residual on
   non-refresh steps.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


class AttnAccelContext:
    """Shared state for tracking cloud denoising steps and acceleration parameters."""

    def __init__(self, r: int = 2, refresh_rate: int = 2):
        self.enabled: bool = False
        self.step: int = 0
        self.r: int = r
        self.refresh_rate: int = refresh_rate
        self.N: int = 4

    def set_step(self, step: int):
        self.step = step

    def reset(self):
        self.step = 0


class I2IAccelAttnProcessor:
    """
    Custom attention processor with per-candidate cross-attention and
    temporal caching on attn2 only.
    """

    def __init__(self, context: AttnAccelContext):
        self.context = context
        self.cached_output: Optional[torch.Tensor] = None
        self.cached_step: int = -1

    def clear_cache(self):
        self.cached_output = None
        self.cached_step = -1

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        # ── Self-attention (attn1): ALWAYS use standard memory-efficient SDPA ──
        if not is_cross_attention:
            return self._sdpa_attn(
                attn, hidden_states, None, attention_mask, temb
            )

        if not self.context.enabled:
            return self._cross_attn_fp32(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )

        current_step = self.context.step
        r = self.context.r

        # ── Cross-attention (attn2), step <= r: compute fresh per-candidate ──
        if current_step <= r:
            return self._cross_attn_fp32(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )

        # ── Cross-attention (attn2), step > r: temporal caching ──────────────
        residual = hidden_states
        should_refresh = (
            self.cached_output is None
            or (current_step - (r + 1)) % self.context.refresh_rate == 0
        )

        if not should_refresh and self.cached_output is not None:
            out = self.cached_output.clone()
            if attn.residual_connection:
                out = out + residual
            out = out / attn.rescale_output_factor
            return out

        # Compute fresh and cache pre-residual output
        return self._compute_and_cache_cross_attn(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, current_step
        )

    # ── Memory-efficient Self-Attention (SDPA) ─────────────────────────

    def _sdpa_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Memory-efficient PyTorch 2.0 SDPA for self-attention (never OOMs)."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    # ── Cross-Attention with FP32 Softmax over text seq_len (77 tokens) ─

    def _cross_attn_fp32(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention with FP32 softmax (seq_len_kv=77, completely safe on memory)."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
        attn_weights = attn_weights.softmax(dim=-1).to(value.dtype)
        hidden_states = torch.matmul(attn_weights, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    # ── Compute + Cache for Cross-Attention ────────────────────────────

    def _compute_and_cache_cross_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
        current_step: int,
    ) -> torch.Tensor:
        """Compute cross-attention, cache the pre-residual output, then return."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
        attn_weights = attn_weights.softmax(dim=-1).to(value.dtype)
        hidden_states = torch.matmul(attn_weights, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        self.cached_output = hidden_states.detach()
        self.cached_step = current_step

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def set_accel_processors(unet, context: AttnAccelContext):
    """Install I2IAccelAttnProcessor across all attention layers of UNet."""
    processors = {}
    for name in unet.attn_processors.keys():
        processors[name] = I2IAccelAttnProcessor(context)
    unet.set_attn_processor(processors)


def restore_vanilla_processors(unet):
    """Restore standard AttnProcessor2_0 across all attention layers."""
    unet.set_attn_processor(AttnProcessor2_0())
