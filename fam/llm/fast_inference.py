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
import tyro
from huggingface_hub import snapshot_download  # type: ignore

from fam.llm.fast_inference_utils import build_model, main
from fam.llm.inference import get_cached_embedding, get_cached_file
from fam.llm.model_decoder import EmbeddingDecoder
from fam.llm.utils import (
    check_audio_file,
    get_default_dtype,
    get_device,
    normalize_text,
)
from fam.telemetry import TelemetryEvent
from fam.telemetry.posthog import PosthogClient

posthog = PosthogClient()  # see fam/telemetry/README.md for more information


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TTS:
    END_OF_AUDIO_TOKEN = 1024

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
        """
        Initialise the TTS model.

        Args:
            model_name: refers to the model identifier from the Hugging Face Model Hub (https://huggingface.co/metavoiceio)
            seed: random seed for reproducibility
            output_dir: directory to save output files
            quantisation_mode: quantisation mode for first-stage LLM.
                Options:
                - None for no quantisation (bf16 or fp16 based on device),
                - int4 for int4 weight-only quantisation,
                - int8 for int8 weight-only quantisation.
            first_stage_path: path to first-stage LLM checkpoint. If provided, this will override the one grabbed from Hugging Face via `model_name`.
            telemetry_origin: A string identifier that specifies the origin of the telemetry data sent to PostHog.
        """

        # NOTE: this needs to come first so that we don't change global state when we want to use
        # the torch.compiled-model.
        self._dtype = get_default_dtype()
        self._device = get_device()
        self._model_dir = snapshot_download(repo_id=model_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if first_stage_path:
            print(f"Overriding first stage checkpoint via provided model: {first_stage_path}")
        self._first_stage_ckpt = first_stage_path or f"{self._model_dir}/first_stage.pt"

        if not os.path.exists(decoder_config_path):
            raise ValueError(f"EmbeddingDecoder config file not found at {decoder_config_path}")

        if not os.path.exists(decoder_checkpoint_file):
            raise ValueError(f"EmbeddingDecoder checkpoint file not found at {decoder_checkpoint_file}")

        with open(decoder_config_path) as f:
            self.decoder_config = AttrDict(json.loads(f.read()))
        self.decoder = EmbeddingDecoder(self.decoder_config).to(self._device)
        state_dict_g = torch.load(decoder_checkpoint_file, map_location=self._device)
        self.decoder.load_state_dict(state_dict_g["generator"])
        self.decoder.eval()
        self.decoder.remove_weight_norm()

        self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        self.model, self.tokenizer, self.smodel, self.model_size = build_model(
            precision=self.precision,
            checkpoint_path=Path(self._first_stage_ckpt),
            spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
            device=self._device,
            compile=True,
            compile_prefill=True,
            quantisation_mode=quantisation_mode,
        )
        self._seed = seed
        self._quantisation_mode = quantisation_mode
        self._model_name = model_name
        self._telemetry_origin = telemetry_origin

    def synthesise(self, text: str, spk_ref_path: str, top_p=0.95, guidance_scale=2.0, temperature=1.0) -> str:
        """
        text: Text to speak
        spk_ref_path: Path to speaker reference file. Min. 30s of audio required. Supports both local paths & public URIs. Audio formats: wav, flac & mp3
        top_p: Top p for sampling applied to first-stage model. Range [0.9, 1.0] are good. This is a measure of speech stability - improves text following for a challenging speaker
        guidance_scale: Guidance scale [1.0, 3.0] for sampling. This is a measure of speaker similarity - how closely to match speaker identity and speech style.
        temperature: Temperature for sampling applied to both LLMs (first & second stage)

        returns: path to speech .wav file
        """
        text = normalize_text(text)
        spk_ref_path = get_cached_file(spk_ref_path)
        check_audio_file(spk_ref_path)
        spk_emb = get_cached_embedding(
            spk_ref_path,
            self.smodel,
        ).to(device=self._device, dtype=self.precision)

        start = time.time()
        # first stage LLM
        _, output_embs = main(
            model=self.model,
            tokenizer=self.tokenizer,
            model_size=self.model_size,
            prompt=text,
            spk_emb=spk_emb,
            top_p=torch.tensor(top_p, device=self._device, dtype=self.precision),
            guidance_scale=torch.tensor(guidance_scale, device=self._device, dtype=self.precision),
            temperature=torch.tensor(temperature, device=self._device, dtype=self.precision),
        )
        # TODO: run EmbeddingDecoder, and save and print output wav_file path?
        output_embs = output_embs.to(dtype=torch.float32).transpose(1, 2)  # (b, c, t)

        model_upsample_factor = math.prod(self.decoder_config.upsample_rates)  # type: ignore
        if self.decoder_config.input_upsampling_factor != model_upsample_factor:  # type: ignore
            output_embs = torch.nn.functional.interpolate(
                output_embs,
                scale_factor=[
                    self.decoder_config.input_upsampling_factor / model_upsample_factor  # type: ignore
                ],  # [320/256] or [160 / 128],
                mode="linear",
            )

        if self.decoder_config.add_noise:  # type: ignore
            output_embs = torch.cat(
                [
                    output_embs,
                    torch.randn(
                        # add model_upsample_factor worth of noise to each input!
                        (output_embs.shape[0], model_upsample_factor, output_embs.shape[-1]),
                        device=output_embs.device,
                        dtype=output_embs.dtype,
                    ),
                ],
                dim=1,
            )

        with torch.no_grad():
            y_g_hat = self.decoder(output_embs)
            audio = y_g_hat.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype("int16")

        wav_file_name = str(Path(self.output_dir) / f"synth_{uuid.uuid4()}.wav")
        scipy.io.wavfile.write(wav_file_name, 24000, audio)
        print(f"\nSaved audio to {wav_file_name}.wav")

        # calculating real-time factor (RTF)
        time_to_synth_s = time.time() - start
        audio, sr = librosa.load(str(wav_file_name))
        duration_s = librosa.get_duration(y=audio, sr=sr)
        real_time_factor = time_to_synth_s / duration_s
        print(f"\nTotal time to synth (s): {time_to_synth_s}")
        print(f"Real-time factor: {real_time_factor:.2f}")

        posthog.capture(
            TelemetryEvent(
                name="user_ran_tts",
                properties={
                    "model_name": self._model_name,
                    "text": text,
                    "temperature": temperature,
                    "guidance_scale": guidance_scale,
                    "top_p": top_p,
                    "spk_ref_path": spk_ref_path,
                    "speech_duration_s": duration_s,
                    "time_to_synth_s": time_to_synth_s,
                    "real_time_factor": round(real_time_factor, 2),
                    "quantisation_mode": self._quantisation_mode,
                    "seed": self._seed,
                    "first_stage_ckpt": self._first_stage_ckpt,
                    "gpu": torch.cuda.get_device_name(0),
                    "telemetry_origin": self._telemetry_origin,
                },
            )
        )

        return str(wav_file_name)


if __name__ == "__main__":
    tts = tyro.cli(TTS)
