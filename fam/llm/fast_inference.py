import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import librosa
import scipy.io.wavfile  # type: ignore
import torch
import torch.multiprocessing
try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except:
    pass

import tyro
from huggingface_hub import snapshot_download  # type: ignore

from fam.llm.fast_inference_utils import build_model, encode_tokens, generate, device_sync
from fam.llm.model_decoder import EmbeddingDecoder
from fam.llm.utils import (
    check_audio_file,
    get_default_dtype,
    get_device,
    normalize_text,
    get_cached_embedding, get_cached_file
)
# from fam.telemetry import TelemetryEvent
# from fam.telemetry.posthog import PosthogClient

# posthog = PosthogClient()  # see fam/telemetry/README.md for more information


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def llm_worker(text_queue: torch.multiprocessing.Queue, embeddings_queue: torch.multiprocessing.Queue, first_stage_ckpt: str, model_dir: str, quantisation_mode):
    # init
    device = "cuda:6"
    dtype = get_default_dtype()
    precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    model, tokeniser, smodel, model_size = build_model(
        precision=precision,
        checkpoint_path=Path(first_stage_ckpt),
        spk_emb_ckpt_path=Path(f"{model_dir}/speaker_encoder.pt"),
        device=device,
        compile=True,
        compile_prefill=True,
        quantisation_mode=quantisation_mode,
    )
    
    # run
    try:
        while True:
            text, spk_ref_path, top_p, guidance_scale, temperature = text_queue.get()
            
            encoded = encode_tokens(tokeniser, text, device=device)
            spk_emb = get_cached_embedding(
                spk_ref_path,
                smodel,
            ).to(device=device, dtype=precision)

            device_sync(device=device)  # MKG
            
            t0 = time.perf_counter()
            for audio_output_embs in generate(
                model=model,
                prompt=encoded,
                spk_emb=spk_emb,
                temperature=torch.tensor(temperature, device=device, dtype=precision),
                top_p=torch.tensor(top_p, device=device, dtype=precision),
                guidance_scale=torch.tensor(guidance_scale, device=device, dtype=precision),
                top_k=None,
                yield_after_k_tokens=24
            ):
                device_sync(device=device)
                audio_output_embs = audio_output_embs.cpu()
                # print(f"llm took: {time.perf_counter() - t0}")

                embeddings_queue.put_nowait(audio_output_embs)

                t0 = time.perf_counter()

            device_sync(device=device)  # MKG

    except KeyboardInterrupt:
        pass
    finally:    
        print("LLMThread stopped")


def decoder_worker(embeddings_queue: torch.multiprocessing.Queue, audio_out_queue: torch.multiprocessing.Queue, decoder_config_path, decoder_checkpoint_file):
    # init
    device = "cuda:7"

    with open(decoder_config_path) as f:
        decoder_config = AttrDict(json.loads(f.read()))
    
    decoder = EmbeddingDecoder(decoder_config).to(device)
    state_dict_g = torch.load(decoder_checkpoint_file, map_location=device)
    decoder.load_state_dict(state_dict_g["generator"])
    decoder.eval()
    decoder.remove_weight_norm()

    # run
    try:
        with torch.no_grad():
            while True:
                output_embs = embeddings_queue.get()

                t0 = time.perf_counter()

                output_embs = output_embs.to(device)
                output_embs = output_embs.to(dtype=torch.float32).transpose(1, 2)  # (b, c, t)
                model_upsample_factor = math.prod(decoder_config.upsample_rates)  # type: ignore
                if decoder_config.input_upsampling_factor != model_upsample_factor:  # type: ignore
                    output_embs = torch.nn.functional.interpolate(
                        output_embs,
                        scale_factor=[
                            decoder_config.input_upsampling_factor / model_upsample_factor  # type: ignore
                        ],  # [320/256] or [160 / 128],
                        mode="linear",
                    )

                if decoder_config.add_noise:  # type: ignore
                    output_embs = torch.cat(
                        [
                            output_embs,
                            torch.randn(
                                # add model_upsample_factor worth of noise to each input!
                                (output_embs.shape[0], model_upsample_factor, output_embs.shape[-1]),
                                device=device,
                                dtype=output_embs.dtype,
                            ),
                        ],
                        dim=1,
                    )

                y_g_hat = decoder(output_embs)
                del output_embs

                audio = y_g_hat.squeeze()
                audio = audio * 32768.0
                audio = audio.cpu().numpy().astype("int16")
                # print(f"decoder took: {time.perf_counter() - t0}")

                audio_out_queue.put_nowait(audio)
    except KeyboardInterrupt:
        pass
    finally:
        print("DecoderThread stopped")


class TTS:
    def __init__(
        self,
        model_name: str = "metavoiceio/metavoice-1B-v0.1",
        decoder_config_path: str = f"{os.path.dirname(os.path.abspath(__file__))}/decoder_config.json",
        decoder_checkpoint_file: str = f"{os.path.dirname(os.path.abspath(__file__))}/decoder.pt",
        *,
        seed: int = 1337,
        output_dir: str = "outputs",
        quantisation_mode: Optional[Literal["int4", "int8"]] = None,
        first_stage_path: Optional[str] = None,
        telemetry_origin: Optional[str] = None,
    ):
        # NOTE: this needs to come first so that we don't change global state when we want to use
        # the torch.compiled-model.
        model_dir = snapshot_download(repo_id=model_name)

        if first_stage_path:
            print(f"Overriding first stage checkpoint via provided model: {first_stage_path}")
        first_stage_ckpt = first_stage_path or f"{model_dir}/first_stage.pt"

        if not os.path.exists(decoder_config_path):
            raise ValueError(f"EmbeddingDecoder config file not found at {decoder_config_path}")

        if not os.path.exists(decoder_checkpoint_file):
            raise ValueError(f"EmbeddingDecoder checkpoint file not found at {decoder_checkpoint_file}")

        # setup queues
        self.text_queue = torch.multiprocessing.Queue()
        self.embeddings_queue = torch.multiprocessing.Queue()
        self.audio_out_queue = torch.multiprocessing.Queue()

        # start processes
        llm_process = torch.multiprocessing.Process(
            target=llm_worker,
            args=(self.text_queue, self.embeddings_queue, first_stage_ckpt, model_dir, quantisation_mode,)
        )
        decoder_process = torch.multiprocessing.Process(
            target=decoder_worker,
            args=(self.embeddings_queue, self.audio_out_queue, decoder_config_path, decoder_checkpoint_file,)
        )

        llm_process.start()
        decoder_process.start()

    def synthesise(self, text: str, spk_ref_path: str, top_p=0.95, guidance_scale=2.0, temperature=1.0) -> str:
        text = normalize_text(text)
        
        spk_ref_path = get_cached_file(spk_ref_path)
        check_audio_file(spk_ref_path)

        self.text_queue.put_nowait((text, spk_ref_path, top_p, guidance_scale, temperature))


if __name__ == "__main__":
    tts = TTS()
