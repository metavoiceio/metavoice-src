import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.decoders import EncodecDecoder
from fam.llm.gptfast_sample_utils import build_model, main
from fam.llm.sample import (
    EncodecDecoder,
    InferenceConfig,
    Model,
    TiltedEncodec,
    TrainedBPETokeniser,
    get_cached_embedding,
    get_enhancer,
)
from fam.llm.utils import check_audio_file, get_default_dtype, normalize_text


class Inferencer:
    def __init__(self):
        # NOTE: this needs to come first so that we don't change global state when we want to use
        # the torch.compiled-model.
        # TODO: check if this works with float16.
        self._dtype = get_default_dtype()
        self._device = "cuda"
        self._model_dir = snapshot_download(repo_id="metavoiceio/metavoice-1B-v0.1")
        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=1024)
        second_stage_ckpt_path = f"{self._model_dir}/second_stage.pt"
        config_second_stage = InferenceConfig(
            ckpt_path=second_stage_ckpt_path,
            num_samples=1,
            seed=1337,
            device=self._device,
            dtype=self._dtype,
            compile=False,
            init_from="resume",
            output_dir=".",
        )
        data_adapter_second_stage = TiltedEncodec(end_of_audio_token=1024)
        self.llm_second_stage = Model(
            config_second_stage, TrainedBPETokeniser, EncodecDecoder, data_adapter_fn=data_adapter_second_stage.decode
        )
        self.enhancer = get_enhancer("df")

        self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        self.model, self.tokenizer, self.smodel, self.model_size = build_model(
            precision=self.precision,
            checkpoint_path=Path(f"{self._model_dir}/first_stage.pt"),
            spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
            device=self._device,
            compile=True,
            compile_prefill=True,
        )

    def synthesize(self, text, spk_ref_path, top_p=0.95, guidance_scale=3.0, temperature=1.0):
        text = normalize_text(text)
        check_audio_file(spk_ref_path)
        spk_emb = get_cached_embedding(
            spk_ref_path,
            self.smodel,
        ).to(device=self._device, dtype=self.precision)
        tokens = main(
            model=self.model,
            tokenizer=self.tokenizer,
            model_size=self.model_size,
            prompt=text,
            spk_emb=spk_emb,
            top_p=torch.tensor(top_p, device=self._device, dtype=self.precision),
            guidance_scale=torch.tensor(guidance_scale, device=self._device, dtype=self.precision),
            temperature=torch.tensor(temperature, device=self._device, dtype=self.precision),
        )
        text_ids, extracted_audio_ids = self.first_stage_adapter.decode([tokens])
        b_speaker_embs = spk_emb.unsqueeze(0)
        wav_files = self.llm_second_stage(
            texts=[text],
            encodec_tokens=[torch.tensor(extracted_audio_ids, dtype=torch.int32, device=self._device).unsqueeze(0)],
            speaker_embs=b_speaker_embs,
            batch_size=1,
            guidance_scale=None,
            top_p=None,
            top_k=200,
            temperature=1.0,
            max_new_tokens=None,
        )
        wav_file = wav_files[0]
        with tempfile.NamedTemporaryFile(suffix=".wav") as enhanced_tmp:
            self.enhancer(str(wav_file) + ".wav", enhanced_tmp.name)
            shutil.copy2(enhanced_tmp.name, str(wav_file) + ".wav")
        return str(wav_file) + ".wav"


if __name__ == "__main__":
    inferencer = Inferencer()
