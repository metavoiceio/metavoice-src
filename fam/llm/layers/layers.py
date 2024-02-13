import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, bias) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.w3 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w3(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.non_linearity = config.nonlinearity_type
        hidden_dim = 4 * config.n_embd
        if config.nonlinearity_type == "gelu":
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        elif config.nonlinearity_type == "swiglu":
            if config.swiglu_multiple_of is None:
                raise Exception("SwiGLU requires swiglu_multiple_of to be set")
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = config.swiglu_multiple_of * math.ceil(hidden_dim / config.swiglu_multiple_of)
            # set name to `c_proj` so that the right initialisation gets applied to it in GPT.__init__()
            self.swiglu = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        else:
            raise Exception(f"Unknown nonlinearity type: {config.nonlinearity_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.non_linearity == "gelu":
            x = self.c_fc(x)
            x = self.gelu(x)
        elif self.non_linearity == "swiglu":
            x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
