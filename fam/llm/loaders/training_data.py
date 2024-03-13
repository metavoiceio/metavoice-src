from pathlib import Path
from typing import Any, Mapping

import julius
import torch
import math
import numpy as np
import pandas as pd
from audiocraft.data.audio import audio_read
from encodec import EncodecModel
from torch.utils.data import DataLoader, Dataset

from fam.llm.fast_inference_utils import encode_tokens
from fam.llm.preprocessing.audio_token_mode import CombinerFuncT, CombinerFuncT
from fam.llm.preprocessing.data_pipeline import pad_tokens
from fam.llm.utils import normalize_text
from fam.quantiser.audio.speaker_encoder.model import SpeakerEncoder
from fam.quantiser.text.tokenise import TrainedBPETokeniser

MBD_SAMPLE_RATE = 24000
ENCODEC_BANDWIDTH = 6


class DynamicComputeDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path | str,
        encodec_model: EncodecModel,
        tokenizer: TrainedBPETokeniser,
        spkemb_model: SpeakerEncoder,
        combiner: CombinerFuncT,
        pad_token: int,
        ctx_window: int,
        device: str,
    ):
        self.dataset_dir = dataset_dir
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.spkemb_model = spkemb_model
        self.device = device
        self.combiner = combiner
        self.pad_token = pad_token
        self.ctx_window = ctx_window
        self.df = pd.read_csv(dataset_dir, delimiter="|", index_col=False)

    @classmethod
    def from_meta(
        cls,
        tokenizer_info: Mapping[str, Any],
        combiner: CombinerFuncT,
        speaker_embedding_ckpt_path: Path | str,
        dataset_dir: Path | str,
        pad_token: int,
        ctx_window: int,
        device: str
    ):
        encodec = EncodecModel.encodec_model_24khz().to(device)
        encodec.set_target_bandwidth(ENCODEC_BANDWIDTH)
        smodel = SpeakerEncoder(
            weights_fpath=str(speaker_embedding_ckpt_path),
            eval=True,
            device=device,
            verbose=False,
        )
        tokeniser = TrainedBPETokeniser(**tokenizer_info)

        return cls(
            dataset_dir,
            encodec,
            tokeniser,
            smodel,
            combiner,
            pad_token,
            ctx_window,
            device
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path, text = self.df.iloc[idx].values.tolist()
        with torch.no_grad():
            text_tokens = self._extract_text_tokens(text)
            encodec_tokens = self._extract_encodec_tokens(audio_path)
            speaker_embedding = self._extract_speaker_embedding(audio_path)
            combined = self.combiner(encodec_tokens, text_tokens)
        padded_combined_tokens = pad_tokens(combined, self.ctx_window, self.pad_token)

        return {"tokens": padded_combined_tokens, "spkemb": speaker_embedding}

    def _extract_text_tokens(self, text: str):
        _text = normalize_text(text)
        _tokens = encode_tokens(self.tokenizer, _text, self.device)

        return _tokens.detach().cpu().numpy()

    def _extract_encodec_tokens(self, audio_path: str):
        wav, sr = audio_read(audio_path)
        if sr != MBD_SAMPLE_RATE:
            wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)

        # Convert to mono and fix dimensionality
        if wav.ndim == 2:
            wav = wav.mean(axis=0, keepdims=True)
        wav = wav.unsqueeze(0)  # Add batch dimension

        wav = wav.to(self.device)
        tokens = self.encodec_model.encode(wav)
        _tokens = tokens[0][0][0].detach().cpu().numpy()  # shape = [8, T]

        return _tokens

    def _extract_speaker_embedding(self, audio_path: str):
        emb = self.spkemb_model.embed_utterance_from_file(audio_path, numpy=False)  # shape = [256,]
        return emb.unsqueeze(0).detach()
