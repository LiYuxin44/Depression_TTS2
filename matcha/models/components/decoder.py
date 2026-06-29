import math
from typing import Optional, Sequence

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from matcha.models.components.transformer import BasicTransformerBlock
from matcha.models.components.adapter import ConvAdapterLayer, AdapterLayer


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.in_channels = dim
        self.conv = torch.nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim)
        self.act = nn.Mish()

    def forward(self, x, mask, film=None):
        h = self.norm(x * mask)
        if film is not None:
            gamma, beta = film
            if gamma is not None:
                h = (1.0 + gamma) * h
            if beta is not None:
                h = h + beta
        h = self.act(h)
        output = self.conv(h)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8, use_adapter=False, adapter_dim=64, adapter_dropout=0.1, adapter_cond_dim: int | None = None,
                 use_decoupled_conditions: bool = False, depression_cond_dim: int | None = None, speaker_cond_dim: int | None = None):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)
        
        # 添加adapter
        self.use_adapter = use_adapter
        self.use_decoupled_conditions = use_decoupled_conditions
        if use_adapter:
            if use_decoupled_conditions:
                self.adapter_dep = ConvAdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=depression_cond_dim or 0)
                self.adapter_spk = ConvAdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=speaker_cond_dim or 0)
            else:
                self.adapter = ConvAdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=adapter_cond_dim)

    def forward(self, x, mask, time_emb, adapter_condition: Optional[torch.Tensor] = None,
                adapter_condition_dep: Optional[torch.Tensor] = None,
                adapter_condition_spk: Optional[torch.Tensor] = None,
                film_params: Optional[tuple] = None):
        film_block1 = film_block2 = None
        if film_params is not None:
            if len(film_params) > 0:
                film_block1 = film_params[0]
            if len(film_params) > 1:
                film_block2 = film_params[1]

        h = self.block1(x, mask, film=film_block1)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask, film=film_block2)
        output = h + self.res_conv(x * mask)
        
        # 应用adapter（带条件）
        if self.use_adapter:
            if self.use_decoupled_conditions:
                if adapter_condition_dep is not None:
                    output = self.adapter_dep(output, condition=adapter_condition_dep)
                if adapter_condition_spk is not None:
                    output = self.adapter_spk(output, condition=adapter_condition_spk)
            else:
                output = self.adapter(output, condition=adapter_condition)
        
        return output


class FilmGenerator(nn.Module):
    def __init__(self, cond_dim: int, film_channels: Sequence[int], hidden_dim: int = 512):
        super().__init__()
        self.cond_dim = cond_dim
        self.film_channels = list(film_channels)
        self.total_channels = sum(self.film_channels)
        if self.total_channels <= 0:
            raise ValueError("film_channels must contain positive values")

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.total_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Optional[torch.Tensor]):
        if cond is None:
            return None
        if cond.dim() == 3 and cond.shape[-1] == 1:
            cond = cond.squeeze(-1)
        if cond.dim() != 2:
            raise ValueError(f"FilmGenerator expects [B, C] condition, got {cond.shape}")

        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)

        outputs = []
        offset = 0
        for channels in self.film_channels:
            g_slice = gamma[:, offset : offset + channels].unsqueeze(-1)
            b_slice = beta[:, offset : offset + channels].unsqueeze(-1)
            outputs.append((g_slice, b_slice))
            offset += channels

        return outputs


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=False,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        use_adapter=False,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_cond_dim: int | None = None,
        use_decoupled_conditions: bool = False,
        depression_cond_dim: int | None = None,
        speaker_cond_dim: int | None = None,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim
        self.adapter_dropout = adapter_dropout
        self.adapter_cond_dim = adapter_cond_dim
        self.use_decoupled_conditions = use_decoupled_conditions
        self.depression_cond_dim = depression_cond_dim or 0
        self.speaker_cond_dim = speaker_cond_dim or 0
        self.film_block_channels: list[int] = []
        self.film_generator: Optional[FilmGenerator] = None

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        # 取消 FiLM 条件注入：不再基于 dep/spk 条件调制 time_mlp
        film_cond_dim = None
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
            cond_proj_dim=film_cond_dim,
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(
                dim=input_channel, 
                dim_out=output_channel, 
                time_emb_dim=time_embed_dim,
                use_adapter=use_adapter,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                adapter_cond_dim=adapter_cond_dim,
                use_decoupled_conditions=use_decoupled_conditions,
                depression_cond_dim=self.depression_cond_dim,
                speaker_cond_dim=self.speaker_cond_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        use_adapter=use_adapter,
                        adapter_dim=adapter_dim,
                        adapter_dropout=adapter_dropout,
                        adapter_cond_dim=adapter_cond_dim,
                        use_decoupled_conditions=use_decoupled_conditions,
                        depression_cond_dim=self.depression_cond_dim,
                        speaker_cond_dim=self.speaker_cond_dim,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
            self._register_resnet_film_channels(resnet)

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(
                dim=input_channel, 
                dim_out=output_channel, 
                time_emb_dim=time_embed_dim,
                use_adapter=use_adapter,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                adapter_cond_dim=adapter_cond_dim,
                use_decoupled_conditions=use_decoupled_conditions,
                depression_cond_dim=self.depression_cond_dim,
                speaker_cond_dim=self.speaker_cond_dim,
            )

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        use_adapter=use_adapter,
                        adapter_dim=adapter_dim,
                        adapter_dropout=adapter_dropout,
                        adapter_cond_dim=adapter_cond_dim,
                        use_decoupled_conditions=use_decoupled_conditions,
                        depression_cond_dim=self.depression_cond_dim,
                        speaker_cond_dim=self.speaker_cond_dim,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))
            self._register_resnet_film_channels(resnet)

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
                use_adapter=use_adapter,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                adapter_cond_dim=adapter_cond_dim,
                use_decoupled_conditions=use_decoupled_conditions,
                depression_cond_dim=self.depression_cond_dim,
                speaker_cond_dim=self.speaker_cond_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        use_adapter=use_adapter,
                        adapter_dim=adapter_dim,
                        adapter_dropout=adapter_dropout,
                        adapter_cond_dim=adapter_cond_dim,
                        use_decoupled_conditions=use_decoupled_conditions,
                        depression_cond_dim=self.depression_cond_dim,
                        speaker_cond_dim=self.speaker_cond_dim,
                    )
                    for _ in range(n_blocks)
                ]
            )

            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True, out_channels=output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
            self._register_resnet_film_channels(resnet)

        self.final_block = Block1D(channels[-1], channels[-1], groups=8)
        self._register_block_film_channels(self.final_block)
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        if self.depression_cond_dim > 0:
            self.film_generator = FilmGenerator(
                cond_dim=self.depression_cond_dim,
                film_channels=self.film_block_channels,
                hidden_dim=max(256, channels[0] * 2),
            )

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(
        block_type,
        dim,
        attention_head_dim,
        num_heads,
        dropout,
        act_fn,
        use_adapter=False,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_cond_dim: int | None = None,
        use_decoupled_conditions: bool = False,
        depression_cond_dim: int | None = None,
        speaker_cond_dim: int | None = None,
    ):
        if block_type == "transformer":
            return BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
                use_adapter=use_adapter,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                adapter_cond_dim=adapter_cond_dim,
                use_decoupled_conditions=use_decoupled_conditions,
                depression_cond_dim=depression_cond_dim,
                speaker_cond_dim=speaker_cond_dim,
            )
        elif block_type == "conformer":
            return ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=dropout,
                ff_dropout=dropout,
                conv_dropout=dropout,
                conv_causal=False,
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def _register_resnet_film_channels(self, resnet_block: ResnetBlock1D):
        self._register_block_film_channels(resnet_block.block1)
        self._register_block_film_channels(resnet_block.block2)

    def _register_block_film_channels(self, block: Block1D):
        if hasattr(block, "in_channels"):
            self.film_block_channels.append(block.in_channels)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model."""
        film_params = None
        film_iter = None
        if self.film_generator is not None:
            film_params = self.film_generator(cond)
            if film_params is not None:
                if len(film_params) != len(self.film_block_channels):
                    raise RuntimeError(
                        f"FiLM params ({len(film_params)}) do not match expected blocks ({len(self.film_block_channels)})."
                    )
                film_iter = iter(film_params)

        def next_film():
            if film_iter is None:
                return None
            return next(film_iter, None)

        def next_film_pair():
            if film_iter is None:
                return None
            first = next_film()
            second = next_film()
            if first is None and second is None:
                return None
            return (first, second)

        t = self.time_embeddings(t)
        # 不再对 time-embedding 做 FiLM 条件注入
        t = self.time_mlp(t)

        # Concatenate the mel spectrogram `x` with the encoder output `mu`
        x = pack([x, mu], "b * t")[0]
        # For multi-speaker: concat speaker embedding along channel dim to match ckpt (e.g., 160+64=224)
        if spks is not None:
            spk_feat = spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # (B, spk_emb_dim, T)
            x = pack([x, spk_feat], "b * t")[0]

        # Now, the rest of the forward pass
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t, film_params=next_film_pair())
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t, film_params=next_film_pair())
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t, film_params=next_film_pair())
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        final_film = next_film()
        x = self.final_block(x, mask_up, film=final_film)
        output = self.final_proj(x * mask_up)

        return output * mask
