import itertools
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from fam.llm.config.finetune_params import audio_token_mode as atm
from fam.llm.config.finetune_params import num_max_audio_tokens_timesteps
from fam.llm.loaders.training_data import DynamicComputeDataset
from fam.llm.preprocessing.audio_token_mode import get_params_for_mode


@pytest.mark.parametrize("dataset", ["tests/resources/datasets/sample_dataset.csv"])
@pytest.mark.skip(reason="Requires ckpt download, not feasible as test suite")
def test_dataset_preprocess_e2e(dataset):
    model_name = "metavoiceio/metavoice-1B-v0.1"
    device = "cuda"
    mode_params = get_params_for_mode(atm, num_max_audio_tokens_timesteps=num_max_audio_tokens_timesteps)

    _model_dir = snapshot_download(repo_id=model_name)
    checkpoint_path = Path(f"{_model_dir}/first_stage.pt")
    spk_emb_ckpt_path = Path(f"{_model_dir}/speaker_encoder.pt")
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=False)
    tokenizer_info = checkpoint.get("meta", {}).get("tokenizer", {})

    dataset = DynamicComputeDataset.from_meta(
        tokenizer_info,
        mode_params["combine_func"],
        spk_emb_ckpt_path,
        dataset,
        mode_params["pad_token"],
        mode_params["ctx_window"],
        device
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    result = next(iter(dataloader))

    # TODO: better assertions based on sample input dims
    assert len(result) == 2
