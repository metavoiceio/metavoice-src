import os
from time import perf_counter as timer
from typing import List, Optional, Union

import librosa
import numpy as np
import torch
from torch import nn

from fam.quantiser.audio.speaker_encoder import audio

mel_window_step = 10
mel_n_channels = 40
sampling_rate = 16000
partials_n_frames = 160
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        weights_fpath: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
        eval: bool = False,
    ):
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        start = timer()

        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if eval:
            self.eval()

        if verbose:
            print("Loaded the speaker embedding model on %s in %.2f seconds." % (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75, numpy: bool = True):
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)  # type: ignore
            partial_embeds = self(mels)

        if numpy:
            partial_embeds = partial_embeds.cpu().numpy()
            raw_embed = np.mean(partial_embeds, axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)
        else:
            raw_embed = partial_embeds.mean(dim=0)
            embed = raw_embed / torch.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def embed_utterance_from_file(self, fpath: str, numpy: bool) -> torch.Tensor:
        wav_tgt, _ = librosa.load(fpath, sr=16000)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        embedding = self.embed_utterance(wav_tgt, numpy=numpy)
        return embedding
