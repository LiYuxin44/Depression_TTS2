from typing import Any, Dict, Optional

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
from diffusers.models.attention import (
    GEGLU,
    GELU,
    AdaLayerNorm,
    AdaLayerNormZero,
    ApproximateGELU,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils.torch_utils import maybe_allow_in_graph
from matcha.models.components.adapter import AdapterLayer


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        use_adapter (`bool`, *optional*, defaults to False): Whether to use adapter.
        adapter_dim (`int`, *optional*, defaults to 64): Adapter dimension.
        adapter_dropout (`float`, *optional*, defaults to 0.1): Adapter dropout.
        adapter_cond_dim (`int`, *optional*, defaults to None): Adapter condition dimension.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_cond_dim: Optional[int] = None,
        use_decoupled_conditions: bool = False,
        depression_cond_dim: Optional[int] = None,
        speaker_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = LoRACompatibleLinear

        self.net = nn.ModuleList([])

        if activation_fn == "snakebeta":
            # 与 ckpt 对齐的顺序和索引：
            # ff.net.0 = SnakeBeta(dim -> inner_dim)  [alpha/beta/proj.*]
            # ff.net.1 = Dropout(dropout)            [无可训练参数]
            # ff.net.2 = Linear(inner_dim -> dim_out) [weight/bias]
            self.net.append(SnakeBeta(dim, inner_dim))
            self.net.append(nn.Dropout(dropout))  # 始终占位在 index 1
            self.net.append(linear_cls(inner_dim, dim_out))
            if final_dropout:
                self.net.append(nn.Dropout(dropout))  # 可选的末尾 dropout（无参数，不影响权重键）
        else:
            # 标准路径：Linear(dim -> inner_dim) -> Dropout -> Activation(inner_dim -> dim_out) -> (Dropout)
            self.net.append(linear_cls(dim, inner_dim))
            self.net.append(nn.Dropout(dropout))
            if activation_fn == "geglu":
                self.net.append(GEGLU(inner_dim, dim_out))
            elif activation_fn == "gelu":
                self.net.append(GELU(inner_dim, dim_out))
            elif activation_fn == "approximate_gelu":
                self.net.append(ApproximateGELU(inner_dim, dim_out))
            else:
                raise ValueError(f"Unknown activation function: {activation_fn}")
            if final_dropout:
                self.net.append(nn.Dropout(dropout))

        # 添加adapter
        self.use_adapter = use_adapter
        self.use_decoupled_conditions = use_decoupled_conditions
        if use_adapter:
            if use_decoupled_conditions:
                self.adapter_dep = AdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=(depression_cond_dim or 0))
                self.adapter_spk = AdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=(speaker_cond_dim or 0))
                self.adapter = None
            else:
                self.adapter = AdapterLayer(dim_out, adapter_dim, adapter_dropout, cond_dim=adapter_cond_dim)
                self.adapter_dep = None
                self.adapter_spk = None

    def forward(self, hidden_states, adapter_condition=None, adapter_condition_dep=None, adapter_condition_spk=None):
        for module in self.net:
            hidden_states = module(hidden_states)
        
        # 应用adapter（带条件）
        if self.use_adapter:
            if self.use_decoupled_conditions and (self.adapter_dep is not None or self.adapter_spk is not None):
                if self.adapter_dep is not None and adapter_condition_dep is not None:
                    hidden_states = self.adapter_dep(hidden_states, condition=adapter_condition_dep)
                if self.adapter_spk is not None and adapter_condition_spk is not None:
                    hidden_states = self.adapter_spk(hidden_states, condition=adapter_condition_spk)
            else:
                hidden_states = self.adapter(hidden_states, condition=adapter_condition)
        
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        use_adapter (`bool`, *optional*, defaults to False): Whether to use adapter.
        adapter_dim (`int`, *optional*, defaults to 64): Adapter dimension.
        adapter_dropout (`float`, *optional*, defaults to 0.1): Adapter dropout.
        adapter_cond_dim (`int`, *optional*, defaults to None): Adapter condition dimension.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_cond_dim: Optional[int] = None,
        use_decoupled_conditions: bool = False,
        depression_cond_dim: Optional[int] = None,
        speaker_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_decoupled_conditions = use_decoupled_conditions

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward（支持双Adapter：由上层调用决定传入哪路条件）
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim, 
            dropout=dropout, 
            activation_fn=activation_fn, 
            final_dropout=final_dropout,
            use_adapter=use_adapter,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            adapter_cond_dim=adapter_cond_dim,
            use_decoupled_conditions=use_decoupled_conditions,
            depression_cond_dim=depression_cond_dim,
            speaker_cond_dim=speaker_cond_dim,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        adapter_condition: Optional[torch.FloatTensor] = None,
        adapter_condition_dep: Optional[torch.FloatTensor] = None,
        adapter_condition_spk: Optional[torch.FloatTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(
                        hid_slice,
                        adapter_condition=adapter_condition,
                        adapter_condition_dep=adapter_condition_dep,
                        adapter_condition_spk=adapter_condition_spk,
                    )
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(
                norm_hidden_states,
                adapter_condition=adapter_condition,
                adapter_condition_dep=adapter_condition_dep,
                adapter_condition_spk=adapter_condition_spk,
            )

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states