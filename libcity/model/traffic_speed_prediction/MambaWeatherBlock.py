"""
MambaWeather: Mamba module with weather condition integration
将天气嵌入拼接到 x_proj 的输入，使动态参数 B, C, Δ 同时依赖输入序列和天气条件
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from libcity.model import loss

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class MambaWeatherBlock(nn.Module):
    """
    Mamba module with weather embedding integration
    改造点：
    - x_proj 输入从卷积后的特征扩展为 [x_conv; w_embed]
    - 动态参数 B, C, Δ 同时依赖输入序列和天气条件
    """
    
    def __init__(
        self,
        d_model=96,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        weather_embed_dim=64,  # 天气嵌入维度
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.weather_embed_dim = weather_embed_dim
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # 关键改造：x_proj 输入维度 = d_inner (卷积输出) + weather_embed_dim (天气嵌入)
        # 输出维度 = dt_rank + d_state * 2 (用于计算 dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner + self.weather_embed_dim,  # 改造：增加天气嵌入维度
            self.dt_rank + self.d_state * 2, 
            bias=False, 
            **factory_kwargs
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # 初始化 dt_proj
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # S4D 初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip" 参数
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # 天气嵌入投影（如果需要对齐维度）
        self.weather_proj = nn.Linear(weather_embed_dim, weather_embed_dim, **factory_kwargs)
        
    def forward(self, hidden_states, weather_embed=None, inference_params=None):
        """
        Args:
            hidden_states: (B, L, D) - 输入序列
            weather_embed: (B, L, D_w) - 天气嵌入，与输入序列对齐，默认为None时自动创建零向量
        Returns:
            same shape as hidden_states: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape
        
        # 如果 weather_embed 为 None，创建零向量作为默认值
        if weather_embed is None:
            print("weather_embed is None, create zero vector as default")
            weather_embed = torch.zeros(
                batch, seqlen, self.weather_embed_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
        
        # 输入投影
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        A = -torch.exp(self.A_log.float())
        
        # 分割 x 和 z
        x, z = xz.chunk(2, dim=1)
        
        # 因果卷积
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        
        # 关键改造：将天气嵌入与卷积输出拼接
        # x: (B, d_inner, L) -> (B, L, d_inner)
        # weather_embed: (B, L, D_w)
        x_transposed = rearrange(x, "b d l -> b l d")  # (B, L, d_inner)
        
        # 投影天气嵌入
        weather_projected = self.weather_proj(weather_embed)  # (B, L, D_w)
        
        # 拼接: [x_conv; w_embed] ∈ R^{B × L × (d_inner + D_w)}
        x_aug = torch.cat([x_transposed, weather_projected], dim=-1)  # (B, L, d_inner + D_w)
        
        # 通过 x_proj 计算 dt, B, C
        x_dbl = self.x_proj(x_aug)  # (B, L, dt_rank + d_state * 2)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt 投影和 reshape
        dt = self.dt_proj.weight @ rearrange(dt, "b l d -> d (b l)")
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        
        # B, C reshape
        B_proj = rearrange(B_proj, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_proj = rearrange(C_proj, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # 选择性扫描
        y = selective_scan_fn(
            x,
            dt,
            A,
            B_proj,
            C_proj,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        
        # 输出投影
        y = rearrange(y, "b d l -> b l d")
        y = self.out_proj(y)
        
        return y


class SimpleMambaWeatherBlock(nn.Module):
    """Simple MambaWeather block with layer normalization and residual connections"""
    
    def __init__(self, d_model=96, d_state=32, d_conv=4, expand=2, weather_embed_dim=None, dropout=0.1):
        super().__init__()
        self.weather_embed_dim = weather_embed_dim
        self.layer_norm = nn.LayerNorm(d_model)
        self.mamba_weather = MambaWeatherBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            weather_embed_dim=weather_embed_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, x, weather_embed=None):
        """
        Args:
            x: (B, L, D) - 输入序列
            weather_embed: (B, L, D_w) - 天气嵌入，默认为None时自动创建零向量
        """
        # 如果 weather_embed 为 None，创建零向量作为默认值
        if weather_embed is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            weather_embed = torch.zeros(
                batch_size, seq_len, self.weather_embed_dim,
                dtype=x.dtype,
                device=x.device
            )
        
        # Mamba block with residual
        residual = x
        x = self.layer_norm(x)
        x = self.mamba_weather(x, weather_embed)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward with residual
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.ff_dropout(x)
        x = x + residual
        
        return x
    
