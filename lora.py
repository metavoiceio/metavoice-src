from torch import nn, Tensor
import torch

from copy import deepcopy

from fam.llm.fast_model import ModelArgs, Transformer

def freeze_parameters_except_lora(model):
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

class LoRALayer(nn.Module):
    def __init__(self, original_dim, adapting_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        
        # Adjust initialization to accommodate original and adapting dimensions
        self.A = nn.Parameter(torch.randn(adapting_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, original_dim))
    
    def forward(self, weight):
        # Compute the low-rank adaptation to the weight
        delta_weight = torch.matmul(self.A, self.B)
        # The addition operation expects the two tensors to be broadcastable;
        # ensure delta_weight's dimensions match those of weight
        return weight + delta_weight

class TransformerWithLoRA(Transformer):
    def __init__(self, config: ModelArgs, device: str = 'cuda'):
        super(TransformerWithLoRA, self).__init__(config)
        
        rank = 10 # Rank value to be tested
        
        # Token embeddings adaptation: [vocab_size, config.dim]
        self.lora_tok_embeddings = LoRALayer(config.dim, config.vocab_size, rank).to(device)

        # Output layer adaptation: [config.dim, vocab_size]
        self.lora_output_weights = LoRALayer(config.vocab_size, config.dim, rank).to(device)
        
        with torch.device(device):
            self.setup_spk_cond_mask()
            self.setup_caches(max_batch_size=2, max_seq_length=config.block_size)
        
    def forward(self, idx: Tensor, spk_emb: Tensor, input_pos: Tensor) -> Tensor:
        mask = self.causal_mask[None, None, input_pos]
        
        # Adjust weights using LoRA
        adjusted_tok_embedding_weights = self.lora_tok_embeddings(self.tok_embeddings.weight)
        adjusted_output_weights = self.lora_output_weights(self.output.weight)
        
        # Use the adjusted weights for operations
        tok_embeddings = torch.nn.functional.embedding(idx, adjusted_tok_embedding_weights)
        
        x = (
            tok_embeddings
            + self.pos_embeddings(input_pos)
            + self.speaker_cond_pos(spk_emb) * self.spk_cond_mask
        )

        for layer in self.layers:
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        
        # Apply adjusted output weights
        logits = torch.nn.functional.linear(x, adjusted_output_weights)

        return logits
    
    @staticmethod
    def from_base_model(base_model: Transformer):
        # Create a new instance of TransformerWithLoRA using the config from the base model
        lora_model = TransformerWithLoRA(base_model.config)
        
        # Here, we manually copy the parameters from the base model to the new model.

        # Since we're integrating LoRA layers specifically, we'll not copy those related parameters
        # but initialize them fresh in the TransformerWithLoRA constructor
        
        # Copy embeddings, layers, and other configurations directly
        lora_model.tok_embeddings = deepcopy(base_model.tok_embeddings)
        lora_model.pos_embeddings = deepcopy(base_model.pos_embeddings)
        lora_model.speaker_cond_pos = deepcopy(base_model.speaker_cond_pos)
        lora_model.layers = deepcopy(base_model.layers)
        lora_model.norm = deepcopy(base_model.norm)
        lora_model.output = deepcopy(base_model.output)
        
        return lora_model