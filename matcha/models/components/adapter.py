import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterLayer(nn.Module):
    """标准的Adapter层，用于在现有模型中插入可训练的参数，支持FiLM条件注入"""
    
    def __init__(self, input_dim, adapter_dim=64, dropout=0.1, cond_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.dropout = dropout
        self.cond_dim = cond_dim
        
        # Adapter结构：down-projection -> activation -> up-projection
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # 条件注入: 生成 FiLM 的 scale / shift
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, adapter_dim * 2)
        else:
            self.cond_proj = None
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化adapter权重"""
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        if self.cond_proj is not None:
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
    
    def forward(self, x, condition=None):
        """
        Args:
            x: 输入张量，shape: (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            condition: 条件张量，shape: (batch_size, cond_dim)
        Returns:
            output: 输出张量，与输入相同shape
        """
        residual = x
        
        # Adapter forward pass
        h = self.down_proj(x)
        
        # FiLM 条件注入（不改变形状）
        if self.cond_proj is not None and condition is not None:
            scale_shift = self.cond_proj(condition)  # (B, 2*adapter_dim)
            gamma, beta = scale_shift.chunk(2, dim=-1)
            # 广播到 (B, T, C) 或 (B, C)
            if h.dim() == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            h = h * (1 + gamma) + beta
        
        h = self.activation(h)
        h = self.dropout_layer(h)
        h = self.up_proj(h)
        
        # Residual connection
        return residual + h


class ConvAdapterLayer(nn.Module):
    """用于卷积层的Adapter，支持FiLM条件注入"""
    
    def __init__(self, input_dim, adapter_dim=64, dropout=0.1, cond_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.dropout = dropout
        self.cond_dim = cond_dim
        
        # 1D卷积版本的adapter
        self.down_proj = nn.Conv1d(input_dim, adapter_dim, kernel_size=1)
        self.up_proj = nn.Conv1d(adapter_dim, input_dim, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # 条件注入: 生成 FiLM 的 scale / shift
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, adapter_dim * 2)
        else:
            self.cond_proj = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        if self.cond_proj is not None:
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
    
    def forward(self, x, condition=None):
        """
        Args:
            x: 输入张量，shape: (batch_size, input_dim, seq_len)
            condition: 条件张量，shape: (batch_size, cond_dim)
        Returns:
            output: 输出张量，与输入相同shape
        """
        residual = x
        
        # Adapter forward pass
        h = self.down_proj(x)
        
        # FiLM 条件注入（不改变形状）
        if self.cond_proj is not None and condition is not None:
            scale_shift = self.cond_proj(condition)  # (B, 2*adapter_dim)
            gamma, beta = scale_shift.chunk(2, dim=-1)
            # 幅度约束：tanh + 小系数
            scale = 0.1
            gamma = torch.tanh(gamma) * scale
            beta = torch.tanh(beta) * scale
            gamma = gamma.unsqueeze(-1)  # (B, C, 1)
            beta = beta.unsqueeze(-1)
            h = h * (1 + gamma) + beta
        
        h = self.activation(h)
        h = self.dropout_layer(h)
        h = self.up_proj(h)
        
        # Residual connection
        return residual + h


class AdapterConfig:
    """Adapter配置类"""
    
    def __init__(
        self,
        use_adapter=True,
        adapter_dim=64,
        adapter_dropout=0.1,
        adapter_layers=None,  # 指定哪些层添加adapter
        freeze_original=True,  # 是否冻结原始参数
        cond_dim: int | None = None,  # 新增：条件维度
    ):
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim
        self.adapter_dropout = adapter_dropout
        self.adapter_layers = adapter_layers or ["decoder"]  # 默认只在decoder添加
        self.freeze_original = freeze_original
        self.cond_dim = cond_dim


def add_adapters_to_model(model, adapter_config):
    """为模型添加adapter层"""
    if not adapter_config.use_adapter:
        return model
    
    # 为decoder添加adapter
    if "decoder" in adapter_config.adapter_layers:
        add_adapters_to_decoder(model.decoder, adapter_config)
    
    # 为encoder添加adapter
    if "encoder" in adapter_config.adapter_layers:
        add_adapters_to_encoder(model.encoder, adapter_config)
    
    return model


def add_adapters_to_decoder(decoder, adapter_config):
    """为decoder添加adapter层"""
    if not hasattr(decoder, 'estimator'):
        return
    
    # 为decoder的estimator添加adapter
    if hasattr(decoder.estimator, 'down_blocks'):
        for i, (resnet, transformer_blocks, downsample) in enumerate(decoder.estimator.down_blocks):
            # 为resnet添加adapter（带cond_dim）
            if hasattr(resnet, 'block1'):
                resnet.adapter = ConvAdapterLayer(
                    resnet.block1.block[0].out_channels,
                    adapter_config.adapter_dim,
                    adapter_config.adapter_dropout,
                    cond_dim=adapter_config.cond_dim,
                )
            
            # 为transformer blocks添加adapter（带cond_dim）
            for transformer in transformer_blocks:
                if hasattr(transformer, 'ff'):
                    transformer.adapter = AdapterLayer(
                        transformer.ff.net[0].out_features,
                        adapter_config.adapter_dim,
                        adapter_config.adapter_dropout,
                        cond_dim=adapter_config.cond_dim,
                    )
    
    # 为mid blocks添加adapter
    for resnet, transformer_blocks in decoder.estimator.mid_blocks:
        if hasattr(resnet, 'block1'):
            resnet.adapter = ConvAdapterLayer(
                resnet.block1.block[0].out_channels,
                adapter_config.adapter_dim,
                adapter_config.adapter_dropout,
                cond_dim=adapter_config.cond_dim,
            )
        
        for transformer in transformer_blocks:
            if hasattr(transformer, 'ff'):
                transformer.adapter = AdapterLayer(
                    transformer.ff.net[0].out_features,
                    adapter_config.adapter_dim,
                    adapter_config.adapter_dropout,
                    cond_dim=adapter_config.cond_dim,
                )
    
    # 为up blocks添加adapter
    for resnet, transformer_blocks, upsample in decoder.estimator.up_blocks:
        if hasattr(resnet, 'block1'):
            resnet.adapter = ConvAdapterLayer(
                resnet.block1.block[0].out_channels,
                adapter_config.adapter_dim,
                adapter_config.adapter_dropout,
                cond_dim=adapter_config.cond_dim,
            )
        
        for transformer in transformer_blocks:
            if hasattr(transformer, 'ff'):
                transformer.adapter = AdapterLayer(
                    transformer.ff.net[0].out_features,
                    adapter_config.adapter_dim,
                    adapter_config.adapter_dropout,
                    cond_dim=adapter_config.cond_dim,
                )


def add_adapters_to_encoder(encoder, adapter_config):
    """为encoder添加adapter层"""
    # 这里可以根据encoder的具体结构添加adapter
    # 由于encoder结构比较复杂，这里先留空
    pass


def freeze_original_parameters(model, adapter_config):
    """冻结原始模型参数，只训练adapter"""
    if not adapter_config.freeze_original:
        return
    
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True 


class FiLMMuAdapter(nn.Module):
    """使用抑郁风格向量对内容序列（如 encoder 对齐后的 mu）进行 FiLM 调制。

    - 输入：
      - content: (B, C, T) 内容序列（通常为 mu）
      - style:   (B, D)   抑郁 embedding（depression embedding）
    - 结构：
      - 风格 MLP：style -> Linear(hidden) -> ReLU -> Dropout -> Linear(2*C)
      - 生成参数：拆分为 gamma、beta
      - 调制：content * (1 + gamma) + beta（在时间维广播）
    - 说明：不对内容序列做投影，MLP 是该适配器中唯一较大的可训练部分
    """

    def __init__(self, channels: int, style_dim: int, mlp_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.mlp_hidden = mlp_hidden

        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2 * channels),
        )

        # 初始化：xavier + 零偏置
        for m in self.style_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """返回调制后的内容序列，shape 与 content 相同。

        Args:
            content: (B, C, T)
            style:   (B, D)
        """
        if style is None:
            return content

        scale_shift = self.style_mlp(style)  # (B, 2C)
        gamma, beta = scale_shift.chunk(2, dim=-1)  # (B, C), (B, C)

        # 广播到时间维
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)    # (B, C, 1)

        return content * (1 + gamma) + beta