import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Type, Union

import librosa
import torch
import tqdm
import tqdm.contrib.concurrent
import tyro
from huggingface_hub import snapshot_download

from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook, TiltedEncodec
from fam.llm.decoders import Decoder, EncodecDecoder
from fam.llm.enhancers import BaseEnhancer, get_enhancer
from fam.llm.model import GPT, GPTConfig
from fam.llm.utils import normalize_text
from fam.quantiser.audio.speaker_encoder.model import SpeakerEncoder
from fam.quantiser.text.tokenise import TrainedBPETokeniser


@dataclass
class InferenceConfig:
    ckpt_path: str  # path to checkpoint
    output_dir: str
    num_samples: int = 10  # number of samples to draw
    seed: int = 1337  # random seed
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    init_from: str = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')

    def __str__(self):
        field_strs = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            field_strs.append(f"  {field.name}: {value}")

        return "InferenceConfig:\n" + "\n".join(field_strs)


class Model:
    def __init__(
        self,
        config: InferenceConfig,
        tokenizer_cls: Type[TrainedBPETokeniser],
        decoder_cls: Type[Decoder],
        data_adapter_fn,
        use_kv_cache: Optional[Literal["none", "flash_decoding", "vanilla"]] = None,
    ):
        # TODO: disentangle the encodec stuff and numbers etc with rest of this code (esp at encoder-only / second stage model inference)
        # TODO: remove magic number
        self._encodec_codes_pad_token = 1024
        self._num_encodec_codebooks = 8
        self.config = config
        self.use_kv_cache = use_kv_cache

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True if config.dtype != "float32" else False  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True if config.dtype != "float32" else False  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in config.device else "cpu"  # for later use in torch.autocast
        self.ptdtype = {
            "float32": torch.float32,
            "tfloat32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.dtype]
        self._ctx = (
            nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        )

        self.use_bpe_tokenizer = False
        self.load_meta = None
        self.speaker_cond = None
        self.meta = None
        self.model = None
        self.checkpoint_config = None
        self.vocab_sizes = None
        self.smodel = None

        self._init_model()

        self.tokenizer = tokenizer_cls(**self.meta["tokenizer"])
        self.decoder = decoder_cls(
            tokeniser_decode_fn=self.tokenizer.decode,
            output_dir=self.config.output_dir,
            data_adapter_fn=data_adapter_fn,
        )

    def _init_model(self):
        if self.config.init_from == "resume":
            # init from a model saved in a specific directory
            checkpoint = torch.load(self.config.ckpt_path, map_location=self.config.device)
            self.vocab_sizes = checkpoint["model_args"]["vocab_sizes"]

            self.load_meta = False
            self.speaker_cond = False

            if "config" in checkpoint:
                self.checkpoint_config = checkpoint["config"]

                self.meta = checkpoint["meta"]
                load_meta = True

            if load_meta:
                self.use_bpe_tokenizer = "stoi" not in self.meta or "itos" not in self.meta
                self.speaker_cond = self.meta.get("speaker_cond")

            if self.speaker_cond:
                speaker_emb_size = self.meta["speaker_emb_size"]

            model_args = checkpoint["model_args"]
            if "causal" in self.checkpoint_config and self.checkpoint_config["causal"] is False:
                self._encodec_ctx_window = model_args["block_size"]

            gptconf = GPTConfig(**model_args)

            # TODO: rename `speaker_emb_dim` to `speaker_emb_size`.
            self.model = GPT(gptconf, speaker_emb_dim=speaker_emb_size if self.speaker_cond else None)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)

        # model
        self.model.eval()
        self.model.to(self.config.device)

        if self.config.compile:
            from einops._torch_specific import allow_ops_in_compiled_graph

            allow_ops_in_compiled_graph()
            self.model = torch.compile(self.model)  # type: ignore

        if self.use_kv_cache is not None:
            if "causal" in self.checkpoint_config and self.checkpoint_config["causal"] is False:
                raise Exception("kv_cache not supported for non-causal models!")

            if self.use_kv_cache == "flash_decoding":
                self.model.enable_kv_cache()
                for block in self.model.transformer.h:
                    block.attn.attn_kernel_type = "fd"
            elif self.use_kv_cache == "vanilla":
                for block in self.model.transformer.h:
                    block.attn.attn_kernel_type = "torch_attn"
                self.model.enable_kv_cache()
            else:
                raise NotImplementedError(f"kv_cache type {self.use_kv_cache} not implemented!")

    def causal_sample(
        self,
        *,
        texts: list[str],
        batch_size: int,
        max_new_tokens: int,
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        speaker_embs: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
    ) -> list[torch.Tensor]:
        """
        Returns list of torch.Tensors of tokens. Each tensor is of shape (1, c, t) where c is the number of codebooks.
        Any flattening / inteleaving / tilting gets reversed before the output is returned.
        """
        if speaker_embs is not None:
            assert len(texts) == len(speaker_embs)

        encoded_texts = [self.tokenizer.encode(text) for text in texts]

        ## create multiple hierarchies and get seq_lens
        seq_lens = []
        xs = []
        for i, encoded_text in enumerate(encoded_texts):
            encoded_text = torch.tensor([encoded_text], dtype=torch.long, device=self.config.device)
            # TODO: remove magic number
            xs.append(
                torch.cat(
                    # [1st hierarchy of text, *remaining hierarchies of padded tokens]
                    # TODO: self.vocab_sizes should be from the model config?
                    [encoded_text, *[torch.ones_like(encoded_text) * 1024] * (len(self.vocab_sizes) - 1)],
                    dim=0,
                ).unsqueeze(0)
            )  # b x [(b=1, c, t)]
            seq_lens.append(xs[-1].shape[-1])
        max_len = max(seq_lens)
        assert len(xs) == len(seq_lens)

        ## equalise the shapes in the batch. we can use torch.zeros as tokens > seq_lens will be masked out.
        x = torch.zeros((len(encoded_texts), xs[0].shape[1], max_len), dtype=torch.long, device=self.config.device)
        for i, _xs in enumerate(xs):
            assert _xs.shape[-1] == seq_lens[i]
            x[i, :, : seq_lens[i]] = _xs

        ## check that the input is correct
        for i in range(x.shape[0]):
            assert x[i, 0, : seq_lens[i]].tolist() == encoded_texts[i]

            # TODO: remove magic number
            if x.shape[1] > 1:
                assert set(x[i, 1, : seq_lens[i]].tolist()) == set([1024])

        assert x.shape[0] == speaker_embs.shape[0] if speaker_embs is not None else True

        if self.speaker_cond is False:
            speaker_embs = None

        # run sampling loop
        with torch.no_grad():
            with self._ctx:  # type: ignore
                to_return = []
                for k in range(self.config.num_samples):
                    assert seq_lens is not None
                    assert batch_size is not None

                    if max(seq_lens) + max_new_tokens >= self.model.config.block_size:
                        raise Exception(
                            f"max_new_tokens {max_new_tokens} too large! Choose {self.model.config.block_size - max(seq_lens) - 1} instead."
                        )

                    y = self.model.generate(
                        x,
                        max_new_tokens,
                        seq_lens=seq_lens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        speaker_embs=speaker_embs,
                        batch_size=batch_size,
                        guidance_scale=guidance_scale,
                        dtype=self.ptdtype,
                        end_of_audio_token=self.tokenizer.offset - 1,
                        end_of_text_token=self.tokenizer.eot_token,
                    )
                    for i in range(len(y)):
                        to_return.append(self.decoder.decode(tokens=y[i].tolist(), causal=True))

                return to_return

    def non_causal_sample(
        self,
        *,
        texts: list[str],
        encodec_tokens: list[torch.Tensor],
        batch_size: int,
        top_k: Optional[int],
        temperature: Optional[float],
        speaker_embs: Optional[torch.Tensor] = None,
    ) -> list[str]:
        """
        Returns paths to saved audio files.
        """
        if speaker_embs is not None:
            assert len(texts) == len(speaker_embs)

        encoded_texts = [self.tokenizer.encode(text) for text in texts]

        # setup input
        # TODO: same code is used during data prep. refactor
        padded_hierarchies_inputs = []
        for encoded_text, encodec_token in zip(encoded_texts, encodec_tokens):
            x = torch.tensor(encoded_text, dtype=torch.long, device=self.config.device)[
                None, None, ...
            ]  # (b=1, c=1, t)

            # TODO: should only happen if decoder is encodecdeocder?
            assert encodec_token.shape[0] == 1
            encodec_token = encodec_token[0].tolist()  # (b=1, c, t) -> (c, t)
            assert len(encodec_token) >= 1 and len(encodec_token) <= self._num_encodec_codebooks

            ## setup hierarchies of tokens
            # TODO: refactor and merge with code in processing.py
            text_tokens = encoded_text  # (t,)

            hierarchies_in = []
            hierarchies_in.append(text_tokens + encodec_token[0] + [self._encodec_codes_pad_token])
            hierarchies_in.append(
                [self._encodec_codes_pad_token] * len(text_tokens) + encodec_token[1] + [self._encodec_codes_pad_token]
            )

            ## adding padding / cutting to the right size as needed
            # TODO: refactor and merge with code in processing.py
            padded_hierarchies_input = []
            for _, t_hierarchy in enumerate(hierarchies_in):
                assert len(t_hierarchy) == len(hierarchies_in[0])
                if len(t_hierarchy) < self._encodec_ctx_window:
                    padded_hierarchies_input.append(
                        t_hierarchy + [self._encodec_codes_pad_token] * (self._encodec_ctx_window - len(t_hierarchy))
                    )
                elif len(t_hierarchy) > self._encodec_ctx_window:
                    padded_hierarchies_input.append(t_hierarchy[: self._encodec_ctx_window])
                else:
                    padded_hierarchies_input.append(t_hierarchy)

            padded_hierarchies_inputs.append(padded_hierarchies_input)

        ## check that the input is correct
        in_x = torch.tensor(padded_hierarchies_inputs, dtype=torch.long, device=self.config.device)
        assert in_x.shape[0] == speaker_embs.shape[0] if speaker_embs is not None else True

        if self.speaker_cond is False:
            speaker_embs = None

        # run sampling loop
        with torch.no_grad():
            with self._ctx:  # type: ignore
                to_return = []
                for k in range(self.config.num_samples):
                    y = self.model.generate(
                        in_x,
                        None,
                        temperature=temperature,
                        top_k=top_k,
                        # TODO: handle separate top_p for this model explicitly
                        top_p=None,
                        speaker_embs=speaker_embs,
                        batch_size=batch_size,
                        guidance_scale=None,
                    )

                    b_tokens = torch.cat([in_x, y], dim=1)
                    for tokens in b_tokens:
                        try:
                            to_return.append(self.decoder.decode(tokens=tokens.tolist(), causal=False))
                        except Exception as e:
                            print("failed to run MBD.")
                            print(f"reason: {str(e)}")
                            to_return.append(None)

                return to_return

    def __call__(
        self,
        *,
        texts: list[str],
        batch_size: int,
        max_new_tokens: Optional[int],
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        encodec_tokens: Optional[list[torch.Tensor]] = None,
        speaker_embs: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
    ):
        if self.checkpoint_config.get("causal", True):
            return self.causal_sample(
                texts=texts,
                batch_size=batch_size,
                speaker_embs=speaker_embs,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            assert encodec_tokens is not None
            assert guidance_scale is None
            assert max_new_tokens is None
            assert top_p is None

            return self.non_causal_sample(
                texts=texts,
                encodec_tokens=encodec_tokens,
                batch_size=batch_size,
                speaker_embs=speaker_embs,
                top_k=top_k,
                temperature=temperature,
            )


def save_result_metadata(wav_path, ref_path, text, first_stage_ckpt_path, second_stage_ckpt_path):
    if first_stage_ckpt_path is None or second_stage_ckpt_path is None:
        return
    json.dump(
        {
            "speaker": ref_path,
            "text": text,
        },
        pathlib.Path(str(wav_path) + ".json").open("w"),
    )


def get_cached_file(file_or_uri: str):
    """
    If it's an s3 file, download it to a local temporary file and return that path.
    Otherwise return the path as is.
    """
    is_uri = file_or_uri.startswith("http")

    cache_path = None
    if is_uri:
        ext = pathlib.Path(file_or_uri).suffix
        # hash the file path to get the cache name
        _cache_name = "audio_" + hashlib.md5(file_or_uri.encode("utf-8")).hexdigest() + ext

        os.makedirs(os.path.expanduser("~/.cache/fam/"), exist_ok=True)
        cache_path = os.path.expanduser(f"~/.cache/fam/{_cache_name}")

        if not os.path.exists(cache_path):
            command = f"curl -o {cache_path} {file_or_uri}"
            subprocess.run(command, shell=True, check=True)
    else:
        if os.path.exists(file_or_uri):
            cache_path = file_or_uri
        else:
            raise FileNotFoundError(f"File {file_or_uri} not found!")

    # check audio file is at min. 30s in length
    audio, sr = librosa.load(cache_path)
    assert librosa.get_duration(y=audio, sr=sr) >= 30, "Speaker reference audio file needs to be >= 30s in duration."

    return cache_path


def get_cached_embedding(local_file_path: str, spkemb_model):
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File {local_file_path} not found!")

    # hash the file path to get the cache name
    _cache_name = "embedding_" + hashlib.md5(local_file_path.encode("utf-8")).hexdigest() + ".pt"

    os.makedirs(os.path.expanduser("~/.cache/fam/"), exist_ok=True)
    cache_path = os.path.expanduser(f"~/.cache/fam/{_cache_name}")

    if not os.path.exists(cache_path):
        spk_emb = spkemb_model.embed_utterance_from_file(local_file_path, numpy=False).unsqueeze(0)  # (b=1, c)
        torch.save(spk_emb, cache_path)
    else:
        spk_emb = torch.load(cache_path)

    return spk_emb


def _sample_utterance_batch(
    texts: list[str],
    spk_cond_paths: list[Optional[str]],
    spkemb_model,
    first_stage_model,
    second_stage_model,
    enhancer: Optional[Union[Literal["df"], BaseEnhancer]],
    first_stage_ckpt_path: str,
    second_stage_ckpt_path: str,
    guidance_scale: Optional[Tuple[float, float]],
    max_new_tokens: int,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    batch_size: int = 128,
) -> List[str]:

    speaker_embs = []
    refs = spk_cond_paths.copy()

    # multithreaded loop to cache all the files
    spk_cond_paths = tqdm.contrib.concurrent.thread_map(
        get_cached_file, spk_cond_paths, desc="getting cached speaker ref files"
    )

    for i, (text, spk_cond_path) in tqdm.tqdm(
        enumerate(zip(texts, spk_cond_paths)), total=len(texts), desc="calculating speaker embeddings"
    ):
        texts[i] = normalize_text(text)
        speaker_embs.append(get_cached_embedding(spk_cond_path, spkemb_model) if spk_cond_path else None)

    b_speaker_embs = torch.cat(speaker_embs, dim=0)
    b_tokens = first_stage_model(
        texts=texts,
        speaker_embs=b_speaker_embs,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # TODO: set batch size for second stage model!
    wav_files = second_stage_model(
        texts=texts,
        encodec_tokens=b_tokens,
        speaker_embs=b_speaker_embs,
        batch_size=batch_size,
        guidance_scale=None,
        top_p=None,
        top_k=top_k,
        temperature=temperature,
        max_new_tokens=None,
    )

    for text, tokens, speaker_embs, ref_name, wav_file in zip(texts, b_tokens, b_speaker_embs, refs, wav_files):
        if wav_file is None:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as enhanced_tmp:
            if enhancer is not None:
                enhancer = get_enhancer(enhancer) if isinstance(enhancer, str) else enhancer
                enhancer(str(wav_file) + ".wav", enhanced_tmp.name)
                # copy enhanced_tmp.name back to wav_file
                print(f"copying enhanced file from {enhanced_tmp.name} to {str(wav_file) + '.wav'}.")
                shutil.copy2(enhanced_tmp.name, str(wav_file) + ".wav")

            save_result_metadata(
                wav_file,
                ref_name,
                text,
                first_stage_ckpt_path,
                second_stage_ckpt_path,
            )
    return [str(w) + ".wav" if not str(w).endswith(".wav") else str(w) for w in wav_files]


def sample_utterance(
    text: str,
    spk_cond_path: Optional[str],
    spkemb_model,
    first_stage_model,
    second_stage_model,
    enhancer: Optional[Union[Literal["df"], BaseEnhancer]],
    first_stage_ckpt_path: str,
    second_stage_ckpt_path: str,
    guidance_scale: Optional[Tuple[float, float]],
    max_new_tokens: int,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
) -> str:
    # NOTE: supports max. 220 characters atm.
    # Long form synthesis coming soon...
    MAX_CHARS = 220
    if len(text) > MAX_CHARS:
        print(
            f"\n***WARNING: Max {MAX_CHARS} characters supported. Provided: {len(text)}. Truncating and generating speech...Can lead to unpredictable speech at the end.***"
        )

    return _sample_utterance_batch(
        texts=[text],
        spk_cond_paths=[spk_cond_path],
        spkemb_model=spkemb_model,
        first_stage_model=first_stage_model,
        second_stage_model=second_stage_model,
        enhancer=enhancer,
        first_stage_ckpt_path=first_stage_ckpt_path,
        second_stage_ckpt_path=second_stage_ckpt_path,
        batch_size=1,
        guidance_scale=guidance_scale,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )[0]


def build_models(config_first_stage, config_second_stage, model_dir, device, use_kv_cache):
    smodel = SpeakerEncoder(
        weights_fpath=os.path.join(model_dir, "speaker_encoder.pt"), device=device, eval=True, verbose=False
    )
    data_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=1024)
    llm_first_stage = Model(
        config_first_stage,
        TrainedBPETokeniser,
        EncodecDecoder,
        data_adapter_fn=data_adapter.decode,
        use_kv_cache=use_kv_cache,
    )
    data_adapter_second_stage = TiltedEncodec(end_of_audio_token=1024)
    llm_second_stage = Model(
        config_second_stage, TrainedBPETokeniser, EncodecDecoder, data_adapter_fn=data_adapter_second_stage.decode
    )
    return smodel, llm_first_stage, llm_second_stage


def get_first_stage_path(model_dir: str):
    """Absolute path to checkpoint for the first stage model."""
    return os.path.join(os.path.expanduser(model_dir), "first_stage.pt")


def get_second_stage_path(model_dir: str):
    """Absolute path to checkpoint for the second stage model."""
    return os.path.join(os.path.expanduser(model_dir), "second_stage.pt")


@dataclass
class SamplingControllerConfig:
    """
    Sample from a trained model.
    """

    huggingface_repo_id: str
    """Absolute path to the model directory."""

    spk_cond_path: str
    """Path to speaker reference file. Min. 30s of audio required. Supports both local paths & public URIs. Audio formats: wav, flac & mp3"""

    text: str = (
        "This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model by MetaVoice."
    )
    """Text to synthesise."""

    num_samples: int = 1
    """Number of samples to generate from each model."""

    max_new_tokens: int = 864
    """Maximum number of new tokens to generate from the first stage model."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    top_k: Optional[int] = None
    """Top k for sampling applied to both models."""

    top_p: Optional[float] = 0.95
    """Top p for sampling applied to first-stage model."""

    seed: int = 1337
    """Random seed for sampling."""

    device: Literal["cuda", "cpu"] = "cuda"
    """Device to use for sampling."""

    dtype: Literal["bfloat16", "float16", "float32", "tfloat32"] = "bfloat16"
    """Data type to use for sampling."""

    compile: bool = False
    """Whether to compile the model using PyTorch 2.0."""

    enhancer: Optional[Literal["df"]] = "df"
    """Enhancer to use for post-processing."""

    init_from: str = "resume"
    """Either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')."""

    use_kv_cache: Optional[Literal["flash_decoding", "vanilla"]] = None
    """Type of kv caching to use for inference: 1) [none] no kv caching, 2) [flash_decoding] use the 
    flash decoding kernel, 3) [vanilla] use flash attention 2 with hand implemented kv-cache."""

    output_dir: str = "samples/"
    """Relative path to output directory"""

    guidance_scale: Optional[Tuple[float, float]] = (3.0, 1.0)
    """Guidance scale for sampling: (speaker conditioning guidance_scale, prompt conditioning guidance scale)."""

    batch_size: int = 128
    """Batch size to use for sampling. Note that the batch size gets doubled when guidance is used. For H100, and 1B model, 
    1 w/ guidance and 1 w/o guidance work well (without kv-caching). With kv-caching, 128 (w/o guidance) and
    64 (w/ guidance) works well."""


if __name__ == "__main__":
    # TODO: add support for batch sampling via CLI. Function has been implemented above.
    sampling_config = tyro.cli(SamplingControllerConfig, use_underscores=True)

    model_dir = snapshot_download(repo_id=sampling_config.huggingface_repo_id)
    first_stage_ckpt_path = get_first_stage_path(model_dir)
    second_stage_ckpt_path = get_second_stage_path(model_dir)

    config_first_stage = InferenceConfig(
        ckpt_path=first_stage_ckpt_path,
        num_samples=sampling_config.num_samples,
        seed=sampling_config.seed,
        device=sampling_config.device,
        dtype=sampling_config.dtype,
        compile=sampling_config.compile,
        init_from=sampling_config.init_from,
        output_dir=sampling_config.output_dir,
    )

    config_second_stage = InferenceConfig(
        ckpt_path=second_stage_ckpt_path,
        num_samples=sampling_config.num_samples,
        seed=sampling_config.seed,
        device=sampling_config.device,
        dtype=sampling_config.dtype,
        compile=sampling_config.compile,
        init_from=sampling_config.init_from,
        output_dir=sampling_config.output_dir,
    )

    sampling_config.max_new_tokens *= (
        2  # deal with max_new_tokens for flattened interleaving! (should scale with num_codebooks?)
    )

    # define models
    smodel, llm_first_stage, llm_second_stage = build_models(
        config_first_stage,
        config_second_stage,
        model_dir=model_dir,
        device=sampling_config.device,
        use_kv_cache=sampling_config.use_kv_cache
    )

    print(f"Synthesising utterance...")
    sample_utterance(
        sampling_config.text,
        os.path.expanduser(sampling_config.spk_cond_path),
        smodel,
        llm_first_stage,
        llm_second_stage,
        sampling_config.enhancer,
        first_stage_ckpt_path,
        second_stage_ckpt_path,
        sampling_config.guidance_scale,
        max_new_tokens=sampling_config.max_new_tokens,
        top_k=sampling_config.top_k,
        top_p=sampling_config.top_p,
        temperature=sampling_config.temperature,
    )
