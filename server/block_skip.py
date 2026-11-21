"""
Block Skip Controller — server/block_skip.py
============================================
Implements UNet Down/Mid-Block Skipping for cloud acceleration with
**interleaved skipping** for t >= s:
  - Even offset (current_step - s) % 2 == 0: run full UNet, refresh cache.
  - Odd offset (current_step - s) % 2 == 1: skip down/mid-blocks, run up-blocks
    with cached residuals.
  - Steps < s: always run full UNet forward pass.
Includes standard post-processing (conv_norm_out -> conv_act -> conv_out).
"""

from typing import Optional, Tuple
import torch


class BlockSkipController:
    """Controls interleaved down-block and mid-block skipping on unet_cloud."""

    def __init__(self, s: int = 3):
        self.enabled: bool = False
        self.s: int = s
        self.current_step: int = 0
        self.cached_mid_sample: Optional[torch.Tensor] = None
        self.cached_down_block_res_samples: Optional[Tuple[torch.Tensor, ...]] = None

    def set_step(self, step: int):
        self.current_step = step

    def reset(self):
        self.current_step = 0
        self.cached_mid_sample = None
        self.cached_down_block_res_samples = None

    def _should_skip(self) -> bool:
        """Determine if this step should skip down/mid-blocks (odd offset from s)."""
        if self.current_step < self.s:
            return False
        if self.cached_mid_sample is None:
            return False  # Need at least one cached pass first
        offset = self.current_step - self.s
        return offset % 2 == 1  # odd offset → skip

    def forward(
        self,
        unet,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs=None,
        attention_mask=None,
    ) -> torch.Tensor:
        """Custom forward pass with interleaved down/mid-block skipping."""
        if not self.enabled:
            return unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
            ).sample

        # Prepare timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = unet.time_proj(timesteps).to(dtype=sample.dtype)
        t_emb = unet.time_embedding(t_emb)

        should_skip = self._should_skip()

        if not should_skip:
            # ── Full Forward Pass (refresh cache if step >= s) ─────────
            sample = unet.conv_in(sample)

            down_block_res_samples = (sample,)
            for downsample_block in unet.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=t_emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=t_emb)
                down_block_res_samples += res_samples

            if unet.mid_block is not None:
                if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
                    sample = unet.mid_block(
                        sample,
                        t_emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample = unet.mid_block(sample, t_emb)

            # Cache at every full-forward step >= s (for interleaved reuse)
            if self.current_step >= self.s:
                self.cached_mid_sample = sample.detach().clone()
                self.cached_down_block_res_samples = tuple(r.detach().clone() for r in down_block_res_samples)

            # Also cache at step s-1 (first entry point)
            if self.current_step == self.s - 1:
                self.cached_mid_sample = sample.detach().clone()
                self.cached_down_block_res_samples = tuple(r.detach().clone() for r in down_block_res_samples)

            # Up-blocks
            for upsample_block in unet.up_blocks:
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=t_emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=t_emb,
                        res_hidden_states_tuple=res_samples,
                    )

            # Post-process
            if getattr(unet, "conv_norm_out", None) is not None:
                sample = unet.conv_norm_out(sample)
                sample = unet.conv_act(sample)
            sample = unet.conv_out(sample)
            return sample

        # ── Interleaved Skip (odd offset from s) ──────────────────────
        sample = self.cached_mid_sample.clone()
        down_res = list(self.cached_down_block_res_samples)

        for upsample_block in unet.up_blocks:
            res_samples = down_res[-len(upsample_block.resnets):]
            down_res = down_res[:-len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                )

        # Post-process
        if getattr(unet, "conv_norm_out", None) is not None:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)
        return sample
