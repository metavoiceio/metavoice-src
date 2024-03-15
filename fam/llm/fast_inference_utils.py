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
import itertools
import time
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import tqdm

from fam.llm.fast_quantize import WeightOnlyInt4QuantHandler, WeightOnlyInt8QuantHandler


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = (
    True  # Experimental feature to reduce compilation times, will be on by default in future
)

# imports need to happen after setting above flags
from fam.llm.fast_model import Transformer
from fam.quantiser.audio.speaker_encoder.model import SpeakerEncoder
from fam.quantiser.text.tokenise import TrainedBPETokeniser


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def top_p_sample(logits: torch.Tensor, top_p: torch.Tensor):
    # ref: huggingface/transformers

    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[-1:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    scores = logits.masked_fill(indices_to_remove, -float("Inf"))
    return scores


def logits_to_probs(
    logits,
    *,
    temperature: torch.Tensor,
    top_p: Optional[torch.Tensor] = None,
    top_k: Optional[torch.Tensor] = None,
):
    logits = logits / torch.max(temperature, 1e-5 * torch.ones_like(temperature))

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    if top_p is not None:
        logits = top_p_sample(logits, top_p)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs


def sample(
    logits,
    guidance_scale: torch.Tensor,
    temperature: torch.Tensor,
    top_p: Optional[torch.Tensor] = None,
    top_k: Optional[torch.Tensor] = None,
):
    # (b, t, vocab_size)
    logits = logits[:, -1]
    logits_cond, logits_uncond_spkemb = logits.split(logits.size(0) // 2, dim=0)
    logits = guidance_scale * logits_cond + (1 - guidance_scale) * logits_uncond_spkemb
    probs = logits_to_probs(logits[0], temperature=temperature, top_p=top_p, top_k=top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    spk_emb: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, spk_emb, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    spk_emb: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, spk_emb, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    spk_emb: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    return_probs: bool = False,
    end_of_audio_token: int = 2048,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in tqdm.tqdm(range(num_new_tokens)):
        if (cur_token == end_of_audio_token).any():
            break
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(model, cur_token, spk_emb, input_pos, **sampling_kwargs)
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            if return_probs:
                new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1).repeat(2, 1)

    return new_tokens, new_probs


def model_forward(model, x, spk_emb, input_pos):
    return model(x, spk_emb, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    spk_emb: torch.Tensor,
    *,
    max_new_tokens: Optional[int] = None,
    callback=lambda x: x,
    end_of_audio_token: int = 2048,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    if max_new_tokens is None:
        max_seq_length = model.config.block_size
    else:
        max_seq_length = T + max_new_tokens
        max_seq_length = min(max_seq_length, model.config.block_size)
    max_new_tokens = max_seq_length - T
    if max_new_tokens <= 0:
        raise ValueError("Prompt is too long to generate more tokens")

    device, dtype = prompt.device, prompt.dtype

    seq = torch.clone(prompt)
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1).repeat(2, 1), spk_emb, input_pos, **sampling_kwargs)
    seq = torch.cat([seq, next_token.view(1)])

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1).repeat(2, 1),
        spk_emb,
        input_pos,
        max_new_tokens - 1,
        callback=callback,
        end_of_audio_token=end_of_audio_token,
        **sampling_kwargs,
    )
    seq = torch.cat([seq, torch.cat(generated_tokens)])

    return seq


def encode_tokens(tokenizer: TrainedBPETokeniser, text: str, device="cuda") -> torch.Tensor:
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(
    checkpoint_path, spk_emb_ckpt_path, device, precision, quantisation_mode: Optional[Literal["int4", "int8"]] = None
):
    ##### MODEL
    with torch.device("meta"):
        model = Transformer.from_name("metavoice-1B")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=False)
    state_dict = checkpoint["model"]
    # convert MetaVoice-1B model weights naming to gptfast naming
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    state_dict["tok_embeddings.weight"] = state_dict.pop("transformer.wtes.0.weight")
    state_dict["pos_embeddings.weight"] = state_dict.pop("transformer.wpe.weight")
    state_dict["output.weight"] = state_dict.pop("lm_heads.0.weight")
    state_dict["norm.weight"] = state_dict.pop("transformer.ln_f.weight")
    for k, v in list(state_dict.items()):
        if k.startswith("transformer.h."):
            state_dict[k.replace("transformer.h.", "layers.")] = state_dict.pop(k)
            k = k.replace("transformer.h.", "layers.")
        if ".attn.c_attn." in k:
            state_dict[k.replace(".attn.c_attn.", ".attention.wqkv.")] = state_dict.pop(k)
            k = k.replace(".attn.c_attn.", ".attention.wqkv.")
        if ".attn.c_proj." in k:
            state_dict[k.replace(".attn.c_proj.", ".attention.wo.")] = state_dict.pop(k)
            k = k.replace(".attn.c_proj.", ".attention.wo.")
        if ".mlp.swiglu.w1." in k:
            state_dict[k.replace(".mlp.swiglu.w1.", ".feed_forward.swiglu.w1.")] = state_dict.pop(k)
            k = k.replace(".mlp.swiglu.w1.", ".feed_forward.swiglu.w1.")
        if ".mlp.swiglu.w3." in k:
            state_dict[k.replace(".mlp.swiglu.w3.", ".feed_forward.swiglu.w3.")] = state_dict.pop(k)
            k = k.replace(".mlp.swiglu.w3.", ".feed_forward.swiglu.w3.")
        if ".ln_1." in k:
            state_dict[k.replace(".ln_1.", ".attention_norm.")] = state_dict.pop(k)
            k = k.replace(".ln_1.", ".attention_norm.")
        if ".ln_2." in k:
            state_dict[k.replace(".ln_2.", ".ffn_norm.")] = state_dict.pop(k)
            k = k.replace(".ln_2.", ".ffn_norm.")
        if ".mlp.c_proj." in k:
            state_dict[k.replace(".mlp.c_proj.", ".feed_forward.w2.")] = state_dict.pop(k)
            k = k.replace(".mlp.c_proj.", ".feed_forward.w2.")

    model.load_state_dict(state_dict, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16)

    if quantisation_mode == "int8":
        warnings.warn(
            "int8 quantisation is slower than bf16/fp16 for undebugged reasons! Please set optimisation_mode to `None` or to `int4`."
        )
        warnings.warn(
            "quantisation will degrade the quality of the audio! Please set optimisation_mode to `None` for best quality."
        )
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        quantized_state_dict = simple_quantizer.create_quantized_state_dict()
        model = simple_quantizer.convert_for_runtime()
        model.load_state_dict(quantized_state_dict, assign=True)
        model = model.to(device=device, dtype=torch.bfloat16)
        # TODO: int8/int4 doesn't decrease VRAM usage substantially... fix that (might be linked to kv-cache)
        torch.cuda.empty_cache()
    elif quantisation_mode == "int4":
        warnings.warn(
            "quantisation will degrade the quality of the audio! Please set optimisation_mode to `None` for best quality."
        )
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize=128)
        quantized_state_dict = simple_quantizer.create_quantized_state_dict()
        model = simple_quantizer.convert_for_runtime(use_cuda=True)
        model.load_state_dict(quantized_state_dict, assign=True)
        model = model.to(device=device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
    elif quantisation_mode is not None:
        raise Exception(f"Invalid quantisation mode {quantisation_mode}! Must be either 'int4' or 'int8'!")

    ###### TOKENIZER
    tokenizer_info = checkpoint.get("meta", {}).get("tokenizer", {})
    tokenizer = TrainedBPETokeniser(**tokenizer_info)

    ###### SPEAKER EMBEDDER
    smodel = SpeakerEncoder(
        weights_fpath=spk_emb_ckpt_path,
        device=device,
        eval=True,
        verbose=False,
    )
    return model.eval(), tokenizer, smodel


def build_model(
    *,
    precision: torch.dtype,
    checkpoint_path: Path = Path(""),
    spk_emb_ckpt_path: Path = Path(""),
    compile_prefill: bool = False,
    compile: bool = True,
    device: str = "cuda",
    quantisation_mode: Optional[Literal["int4", "int8"]] = None,
):
    assert checkpoint_path.is_file(), checkpoint_path

    print(f"Using device={device}")

    print("Loading model ...")
    t0 = time.time()
    model, tokenizer, smodel = _load_model(
        checkpoint_path, spk_emb_ckpt_path, device, precision, quantisation_mode=quantisation_mode
    )

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])

    with torch.device(device):
        model.setup_spk_cond_mask()
        model.setup_caches(max_batch_size=2, max_seq_length=model.config.block_size)

    if compile:
        print("Compiling...Can take up to 2 mins.")
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token,
            mode="max-autotune",
            fullgraph=True,
        )

        if compile_prefill:
            prefill = torch.compile(
                prefill,
                fullgraph=True,
                dynamic=True,
            )

    encoded = encode_tokens(tokenizer, "Hello, what's up?", device=device)
    spk_emb = torch.randn((1, 256), device=device, dtype=precision)

    device_sync(device=device)  # MKG
    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        spk_emb,
        max_new_tokens=200,
        callback=lambda x: x,
        temperature=torch.tensor(1.0, device=device, dtype=precision),
        top_k=None,
        top_p=torch.tensor(0.95, device=device, dtype=precision),
        guidance_scale=torch.tensor(3.0, device=device, dtype=precision),
        end_of_audio_token=9999,  # don't end early for compilation stage.
    )

    device_sync(device=device)  # MKG

    print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

    return model, tokenizer, smodel, model_size


def main(
    *,
    model,
    tokenizer,
    model_size,
    prompt: str,
    guidance_scale: torch.Tensor,
    temperature: torch.Tensor,
    spk_emb: torch.Tensor,
    top_k: Optional[torch.Tensor] = None,
    top_p: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> list:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""

    encoded = encode_tokens(tokenizer, prompt, device=device)
    prompt_length = encoded.size(0)

    aggregate_metrics: dict = {
        "tokens_per_sec": [],
    }

    device_sync(device=device)  # MKG

    if True:
        callback = lambda x: x
    t0 = time.perf_counter()

    y = generate(
        model,
        encoded,
        spk_emb,
        callback=callback,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        guidance_scale=guidance_scale,
    )

    device_sync(device=device)  # MKG
    t = time.perf_counter() - t0

    tokens_generated = y.size(0) - prompt_length
    tokens_sec = tokens_generated / t
    aggregate_metrics["tokens_per_sec"].append(tokens_sec)
    print(f"Time for 1st stage LLM inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
    print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    # print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB\n")

    return y.tolist()
