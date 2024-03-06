from torch import nn, Tensor
import torch

from copy import deepcopy

from fam.llm.fast_model import ModelArgs, Transformer
from torch.nn import functional as F
import math

def freeze_parameters_except_lora(model):
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        else:
            print(f"LoRA parameter: {name}")
            param.requires_grad = True

# LoRALinear adapted from GitHub nanoGPT-LoRA:
# https://github.com/danielgrittner/nanoGPT-LoRA/blob/master/model.py#L20
# Overwriting the methods of nn.Linear:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class LoRALinear(nn.Linear):
    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        # LoRA stuff
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / lora_rank
            self.lora_A = nn.Parameter(torch.empty((lora_rank, self.in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Same as nn.Linear
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged and self.is_lora():
            # h = Wx + BAx * scaling
            x += self.lora_scaling * F.linear(
                F.linear(
                    self.lora_dropout(input),
                    self.lora_A
                ),
                self.lora_B
            )
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged and self.is_lora():
            # de-merge weights, i.e., remove BA from W = W + BA
            self.weight.data -= self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged and self.is_lora():
            # merge weights, i.e., add BA to W
            self.weight.data += self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = True
        return self


class TransformerWithLoRA(Transformer):
    def __init__(self, config: ModelArgs, device: str = 'cuda'):
        super(TransformerWithLoRA, self).__init__(config)

        # LoRALinear parameters for the speaker_cond_pos layer
        self.speaker_cond_pos = LoRALinear(
            in_features=config.speaker_emb_dim,
            out_features=config.dim,
            bias=False,
            lora_rank=16, # Test
            lora_alpha=0.5, # Test
            lora_dropout=0.1, # Test
        )
        
        with torch.device(device):
            self.setup_spk_cond_mask()
            self.setup_caches(max_batch_size=2, max_seq_length=config.block_size)
            
    def forward(self, idx: Tensor, spk_emb: Tensor, input_pos: Tensor, targets: Tensor = None) -> Tensor:
        mask = self.causal_mask[None, None, input_pos]
        
        x = (
            self.tok_embeddings(idx)
            + self.pos_embeddings(input_pos)
            + self.speaker_cond_pos(spk_emb) * self.spk_cond_mask
        )

        for layer in self.layers:
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)

        if targets is not None:
            # logits is (B, T, V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss

        return logits
    
    @staticmethod
    def from_base_model(base_model: Transformer, precision: torch.dtype, device: str = 'cuda'):
        # Create a new instance of TransformerWithLoRA using the config from the base model
        lora_model = TransformerWithLoRA(base_model.config)
        
        # Copy embeddings, layers, and other configurations directly
        lora_model.tok_embeddings = deepcopy(base_model.tok_embeddings)
        lora_model.pos_embeddings = deepcopy(base_model.pos_embeddings)
        lora_model.speaker_cond_pos = deepcopy(base_model.speaker_cond_pos)
        lora_model.layers = deepcopy(base_model.layers)
        lora_model.norm = deepcopy(base_model.norm)
        lora_model.output = deepcopy(base_model.output)

        # Now set the speaker_cond_pos layer to use LoRALinear but with the preloaded non-LoRA weights
        lora_model.speaker_cond_pos = LoRALinear(
            in_features=base_model.config.speaker_emb_dim,
            out_features=base_model.config.dim,
            bias=False,
            lora_rank=16, # Test
            lora_alpha=0.5, # Test
            lora_dropout=0.1, # Test
        )
        lora_model.speaker_cond_pos.weight = base_model.speaker_cond_pos.weight
        
        # Set the device
        with torch.device(device):
            lora_model.setup_spk_cond_mask()
            lora_model.setup_caches(max_batch_size=2, max_seq_length=base_model.config.block_size)
        
        lora_model = lora_model.to(device=device, dtype=precision)
        
        return lora_model