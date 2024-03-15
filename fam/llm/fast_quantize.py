# Copyright (c) MetaVoice Labs Inc., Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

default_device = "cuda" if torch.cuda.is_available() else "cpu"

##### Quantization Primitives ######


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int).to(torch.int32).reshape_as(w)

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def group_dequantize_tensor_from_qparams(w_int32, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int32.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int32.shape[-1]
    assert w_int32.shape[-1] % groupsize == 0
    assert w_int32.dim() == 2

    w_int32_grouped = w_int32.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = w_int32_grouped.sub(2 ** (n_bit - 1)).mul(scales).add(zeros).reshape_as(w_int32)
    return w_dq


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            # TODO: quantise RMSNorm as well.
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, torch.int8)
                cur_state_dict[f"{fqn}.weight"] = int8_weight.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype).to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


##### weight only int4 per channel groupwise quantized code ######


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(weight_bf16, n_bit=4, groupsize=groupsize)
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


def replace_linear_int4(module, groupsize, inner_k_tiles, padding, use_cuda):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.out_features % 8 == 0:
            if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles):
                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        groupsize=groupsize,
                        inner_k_tiles=inner_k_tiles,
                        padding=False,
                        use_cuda=use_cuda,
                    ),
                )
            elif padding:
                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        groupsize=groupsize,
                        inner_k_tiles=inner_k_tiles,
                        padding=True,
                        use_cuda=use_cuda,
                    ),
                )
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding, use_cuda)


class WeightOnlyInt4QuantHandler:
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        self.mod = mod
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                if out_features % 8 != 0:
                    continue
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding:
                        import torch.nn.functional as F
                        from model import find_multiple

                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")

        return cur_state_dict

    def convert_for_runtime(self, use_cuda):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding, use_cuda)
        return self.mod


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        padding: bool = True,
        use_cuda=True,
    ) -> None:
        super().__init__()
        self.padding = padding
        if padding:
            from model import find_multiple

            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        if use_cuda:
            self.register_buffer(
                "weight",
                torch.empty(
                    (out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32
                ),
            )
        else:
            self.register_buffer("weight", torch.empty((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer(
            "scales_and_zeros", torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F

            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)
