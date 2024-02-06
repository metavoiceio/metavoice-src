import torch.nn as nn

from fam.llm.layers.attn import SelfAttention
from fam.llm.layers.layers import MLP, LayerNorm, RMSNorm


class Block(nn.Module):
    """
    Block class represents a single block in the model.

    Args:
        config (object): Configuration object containing parameters for the block.

    Attributes:
        ln_1 (object): Layer normalization for the attention layer.
        ln_2 (object): Layer normalization for the feed-forward layer.
        attn (object): Self-attention layer.
        mlp (object): Multi-layer perceptron layer.

    Methods:
        forward(x): Performs forward pass through the block.
    """

    def __init__(self, config):
        super().__init__()
        if config.norm_type == "rmsnorm":
            if config.rmsnorm_eps is None:
                raise Exception("RMSNorm requires rmsnorm_eps to be set")
            self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)  # attn norm
            self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)  # ffn norm
        elif config.norm_type == "layernorm":
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # attn norm
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # ffn norm
        else:
            raise Exception(f"Unknown norm type: {config.norm_type}")
        self.attn = SelfAttention(config)

        self.mlp = MLP(config)

    def forward(self, x):
        """
        Performs forward pass through the block.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after passing through the block.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
