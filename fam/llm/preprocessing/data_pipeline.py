from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np


def pad_tokens(tokens: np.ndarray, context_window: int, pad_token: int) -> np.ndarray:
    """Pads or truncates a single example to the context_window + 1 size.

    tokens: (..., example_length)
    """
    example_length = tokens.shape[-1]
    if example_length > context_window + 1:
        # Truncate
        tokens = tokens[..., : context_window + 1]
    elif example_length < context_window + 1:
        # Pad
        padding = np.full(tokens.shape[:-1] + (context_window + 1 - example_length,), pad_token)
        tokens = np.concatenate([tokens, padding], axis=-1)
    assert tokens.shape[-1] == context_window + 1
    return tokens


def get_training_tuple(
    batch: Dict[str, Any],
    causal: bool,
    num_codebooks: Optional[int],
    speaker_cond: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # batch contains combined tokens as specified by audio_token_mode
    if causal:
        num_codebooks = batch["tokens"].shape[1] if num_codebooks is None else num_codebooks
        x = batch["tokens"][:, :num_codebooks, :-1]
        y = batch["tokens"][:, :num_codebooks, 1:]

    se = batch["spkemb"]

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    se = se.to(device, non_blocking=True) if speaker_cond else None

    return x, y, se


def pad_with_values(tensor, batch_size, value):
    """Pads the tensor up to batch_size with values."""
    if tensor.shape[0] < batch_size:
        return torch.cat(
            [
                tensor,
                torch.full(
                    (batch_size - tensor.shape[0], *tensor.shape[1:]), value, dtype=tensor.dtype, device=tensor.device
                ),
            ]
        )
    else:
        return tensor
