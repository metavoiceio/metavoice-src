import pathlib

import julius
import torch
import typing as tp
from audiocraft.data.audio import audio_read
from encodec import EncodecModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook

from fam.llm.fast_inference_utils import encode_tokens
from fam.llm.inference import SpeakerEncoder, TrainedBPETokeniser, get_cached_embedding
from fam.llm.utils import normalize_text

MBD_SAMPLE_RATE = 24000
END_OF_AUDIO_TOKEN = 1024

class MetavoiceDataset(Dataset):
    def __init__(self, dataset_dir: str, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str):
        
        self.dataset_dir = dataset_dir
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.spkemb_model = spkemb_model
        self.device = device

        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=END_OF_AUDIO_TOKEN)

        # Loop through dataset_dir and create a list of tuples (wav_path, text)
        # File system will look like:
        # dataset_dir/<utt_id>.wav and dataset_dir/<utt_id>.txt
        self.data_list = []
        for audio_file in pathlib.Path(dataset_dir).glob('*.wav'):
            utt_id = audio_file.stem
            wav_path = f"{dataset_dir}/{utt_id}.wav"
            txt_path = f"{dataset_dir}/{utt_id}.txt"
            with open(txt_path, 'r') as f:
                text = f.read()
            
            self.data_list.append((wav_path, text))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, text = self.data_list[idx]

        # Extract text tokenization
        prompt = self._extract_text_tokens(text)

        # Extract encodec tokens
        encodec_tokens = self._extract_encodec_tokens(audio_path)

        # Extract speaker embeddings
        spk_emb = self._extract_speaker_embeddings(audio_path)

        return prompt, encodec_tokens, spk_emb

    def _extract_text_tokens(self, text: str):
        # For text tokens, one can use the tokenizer per:
        # https://github.com/metavoiceio/metavoice-src/blob/main/fam/llm/inference.py#L177
        text = normalize_text(text)
        encoded = encode_tokens(self.tokenizer, text, device=self.device)

        return encoded

    def _extract_encodec_tokens(self, audio_path: str):
        # read audio
        wav, sr = audio_read(audio_path)

        # Resample to MBD's expected sample rate
        if sr != MBD_SAMPLE_RATE:
            wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)

        # Convert to mono and fix dimensionality
        if wav.ndim == 2:
            wav = wav.mean(axis=0, keepdims=True)
        wav = wav.unsqueeze(0)  # Add batch dimension

        # Extract tokens
        wav = wav.to(self.device)
        tokens = self.encodec_model.encode(wav) # list[EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]]

        tokens = tokens[0][0][0] # (8, T)

        # Only return tokens in first 2 hierarchies for training stage 1
        # Not sure if this is the correct approach.
        tokens = tokens[:2] # (2, T)

        # Interleave and flatten the first two hierarchies
        # Then multiply every 2nd token by 2 to make space for the second hierarchy
        tokens = tokens.flatten() # (2*T)
        tokens[1::2] *= 2

        # Convert tokens to list before decoding to audio indices
        tokens = tokens.tolist() # list[int]

        # convert into audio ids
        _, extracted_audio_ids = self.first_stage_adapter.decode([tokens])

        # list[list[int], list[int]] -> (2, T), dtype long
        encodec_tokens = torch.tensor(extracted_audio_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Interleave tokens and flatten (2, T) -> (2T,)
        encodec_tokens = encodec_tokens.flatten() # (2T,)

        return encodec_tokens # (2T,)

    def _extract_speaker_embeddings(self, audio_path: str):
        # For speaker embedding, you can also follow the code at:
        # https://github.com/metavoiceio/metavoice-src/blob/main/fam/llm/inference.py#L435
        return get_cached_embedding(audio_path, self.spkemb_model)

def custom_collate_fn(batch):
    # Unzip the batch
    prompts, encodec_tokens, spk_embds = zip(*batch)

    B = prompts.shape[0]
    P = prompts.shape[1]
    T = encodec_tokens.shape[-1]
    T_rand = torch.randint(0, T, (1,)).item()

    S = P + T_rand

    # X -> (S)
    X = torch.cat((prompt, encodec_tokens[:T_rand]), dim=-1)
    
    
    # input_pos -> (S)
    input_pos = torch.arange(S, device=self.device, dtype=torch.long)
    
    # Y -> (S)
    Y = torch.cat((prompt, encodec_tokens[1:T_rand+1]), dim=-1)

    return X, Y, spk_emb, input_pos

def create_metavoice_dataloaders(dataset_dir: str, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str, batch_size: int = 16, validation_split: float = 0.2, shuffle: bool = True):
    full_dataset = MetavoiceDataset(dataset_dir, encodec_model, tokenizer, spkemb_model, device)
    
    # Split dataset into training and validation
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

    return train_loader, validation_loader