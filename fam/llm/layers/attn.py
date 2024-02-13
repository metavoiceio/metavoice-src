import math
import warnings

import torch
import torch.nn as nn

try:
    from flash_attn import (  # type: ignore
        flash_attn_func,
        flash_attn_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    warnings.warn("flash_attn not installed, make sure to replace attention mechanism with torch_attn")
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, config):
        """
        Initializes the SelfAttention module.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.
        """
        super().__init__()
        self._validate_config(config)
        self._initialize_parameters(config)

    def empty_kv_cache(self, batch_size: int, kv_cache_maxlen: int, dtype: torch.dtype):
        """
        Empties the key-value cache.

        Args:
            batch_size: The batch size.
            kv_cache_maxlen: The maximum length of the key-value cache.
            dtype: The data type of the cache.

        Raises:
            Exception: If trying to empty the KV cache when it is disabled.
        """
        if self.kv_cache_enabled is False:
            raise Exception("Trying to empty KV cache when it is disabled")

        # register so that the cache moves devices along with the module
        # TODO: get rid of re-allocation.
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                2,
                batch_size,
                kv_cache_maxlen,
                self.n_head,
                self.n_embd // self.n_head,
                dtype=dtype,
                device=self.c_attn.weight.device,
            ),
            persistent=False,
        )

        self.kv_cache_first_empty_index = 0

    def _initialize_parameters(self, config):
        """
        Initializes the parameters of the SelfAttention module.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.
        """
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal
        self.attn_kernel_type = config.attn_kernel_type
        self.attn_dropout = nn.Dropout(config.dropout)

        self.kv_cache_enabled = False

    def _validate_config(self, config):
        """
        Validates the configuration parameters.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.

        Raises:
            AssertionError: If the embedding dimension is not divisible by the number of heads.
        """
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

    def _update_kv_cache(self, q, k, v):
        """
        Updates the key-value cache.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.

        Returns:
            The updated key and value tensors.

        Raises:
            AssertionError: If the dimensions of the query, key, and value tensors are not compatible.
        """
        q_time, k_time, v_time = q.shape[1], k.shape[1], v.shape[1]

        if self.kv_cache_first_empty_index == 0:
            assert q_time == k_time and q_time == v_time
        else:
            assert (
                q_time == 1
            ), f"Only one query at a time is supported, but got q_time={q_time} for kv_cache_first_empty_index={self.kv_cache_first_empty_index}"

        self.kv_cache[0, :, self.kv_cache_first_empty_index : self.kv_cache_first_empty_index + q_time] = k
        self.kv_cache[1, :, self.kv_cache_first_empty_index : self.kv_cache_first_empty_index + q_time] = v
        self.kv_cache_first_empty_index += q_time

        k = self.kv_cache[0, :, : self.kv_cache_first_empty_index]
        v = self.kv_cache[1, :, : self.kv_cache_first_empty_index]

        return k, v

    def _fa2_attention(self, c_x: torch.Tensor) -> torch.Tensor:
        """
        Performs Flash Attention 2.0 CUDA kernel based attention.

        Args:
            c_x: The input tensor.

        Returns:
            The output tensor.
        """
        if self.kv_cache_enabled:
            q, k, v = c_x.split(1, dim=2)
            q = q.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)

            k, v = self._update_kv_cache(q, k, v)

            y = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0,
                softmax_scale=None,
                causal=self.causal,
            )
        else:
            # efficient attention using Flash Attention 2.0 CUDA kernels
            y = flash_attn_qkvpacked_func(
                c_x, dropout_p=self.dropout if self.training else 0, softmax_scale=None, causal=self.causal
            )  # outputs (B, T, nh, hs)

        return y

    def _fd_attention(self, c_x: torch.Tensor) -> torch.Tensor:
        """
        Performs Flash decoding based attention.

        Args:
            c_x: The input tensor.

        Returns:
            The output tensor.

        Raises:
            Exception: If key-value caching is not enabled.
            Exception: If non-causal attention is activated.
        """
        if self.kv_cache_enabled is False:
            raise Exception("Flash decoding required kv_cache to be enabled")

        if self.causal is False:
            raise Exception("It is only supported for causal attention")

        q, k, v = c_x.split(1, dim=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        y = flash_attn_with_kvcache(
            q,
            self.kv_cache[0],
            self.kv_cache[1],
            k,
            v,
            cache_seqlens=self.kv_cache_first_empty_index,
            softmax_scale=None,
            causal=self.causal,
        )
        self.kv_cache_first_empty_index += q.shape[1]

        return y

    def _torch_attn(self, c_x: torch.Tensor) -> torch.Tensor:
        """
        Performs attention using the torch.nn.functional.scaled_dot_product_attention function.

        Args:
            c_x: The input tensor.

        Returns:
            The output tensor.
        """
        q, k, v = c_x.split(1, dim=2)  # q, k, v of shape (B, T, 1, nh, hs)
        q = q.squeeze(2)  # (B, T, nh, hs)
        k = k.squeeze(2)  # (B, T, nh, hs)
        v = v.squeeze(2)  # (B, T, nh, hs)

        if self.kv_cache_enabled:
            k, v = self._update_kv_cache(q, k, v)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.causal and (not self.kv_cache_enabled or self.kv_cache_first_empty_index == 0),
        ).transpose(
            1, 2
        )  # (B, nh, T, hs) -> (B, T, nh, hs)

        return y

    def _vanilla_attn(self, c_x: torch.Tensor) -> torch.Tensor:
        """
        Performs vanilla attention.

        Args:
            c_x: The input tensor.

        Returns:
            The output tensor.
        """
        q, k, v = c_x.split(1, dim=2)  # q, k, v of shape (B, T, nh, hs)
        q = q.squeeze(2)  # (B, T, nh, hs)
        k = k.squeeze(2)  # (B, T, nh, hs)
        v = v.squeeze(2)  # (B, T, nh, hs)

        if self.kv_cache_enabled:
            k, v = self._update_kv_cache(q, k, v)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        if self.causal and (not self.kv_cache_enabled or self.kv_cache_first_empty_index == 0):
            att = att.masked_fill(
                torch.triu(torch.ones_like(att, dtype=torch.bool), diagonal=1), float("-inf")
            )  # (B, nh, T, T)
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_dropout(att)  # (B, nh, T, T)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2)  # (B, T, nh, hs)

        return y

    def forward(self, x):
        """
        Performs the forward pass of the SelfAttention module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_x = self.c_attn(x).view(B, T, 3, self.n_head, C // self.n_head)  # (B, T, 3, nh, hs)

        # causal self-attention;
        if self.attn_kernel_type == "fa2":
            y = self._fa2_attention(c_x)
        elif self.attn_kernel_type == "fd":
            y = self._fd_attention(c_x)
        elif self.attn_kernel_type == "torch_attn":
            y = self._torch_attn(c_x)
        elif self.attn_kernel_type == "hand":
            y = self._vanilla_attn(c_x)
        else:
            raise Exception(f"Unknown attention kernel type: {self.attn_kernel_type}")

        y = y.contiguous().view(B, T, C)  # re-assemble all head outputs side by side: (B, T, nh, hs) -> (B, T, hs * nh)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
