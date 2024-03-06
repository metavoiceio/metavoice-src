import pathlib

import julius
import torch
from audiocraft.data.audio import audio_read
from encodec import EncodecModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from fam.llm.fast_inference_utils import encode_tokens
from fam.llm.inference import SpeakerEncoder, TrainedBPETokeniser, get_cached_embedding
from fam.llm.utils import normalize_text

MBD_SAMPLE_RATE = 24000

class MetavoiceDataset(Dataset):
    def __init__(self, dataset_dir: str, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str):
        
        self.dataset_dir = dataset_dir
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.spkemb_model = spkemb_model
        self.device = device

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
        text_tokens = self._extract_text_tokens(text)

        # Extract encodec tokens
        encodec_tokens = self._extract_encodec_tokens(audio_path)

        # Extract speaker embedding
        speaker_embedding = self._extract_speaker_embeddings(audio_path)

        # Some of these fields may be redundant, useful for testing right now.
        return text_tokens, encodec_tokens, speaker_embedding

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

        tokens = tokens[0][0][0] # shape (8, T)
        print("tokens2 shape ", tokens.shape)

        # Only return tokens in first 2 hierarchies for training stage 1
        # Not sure if this is the correct approach.
        tokens = tokens[:2] # shape (2, T)
        print("tokens3 shape ", tokens.shape) # shape (2, T)

        # Interleave and flatten the first two hierarchies
        tokens = tokens.transpose(0, 1).flatten() # (T, 2) -> (2T)
        print("tokens4 shape ", tokens.shape)
        
        return tokens

    def _extract_speaker_embeddings(self, audio_path: str):
        # For speaker embedding, you can also follow the code at:
        # https://github.com/metavoiceio/metavoice-src/blob/main/fam/llm/inference.py#L435
        return get_cached_embedding(audio_path, self.spkemb_model)

def custom_collate_fn(batch):
    text_tokens, encodec_tokens, speaker_embeddings = zip(*batch)
    
    # Padding for text tokens
    text_tokens = pad_sequence(text_tokens, batch_first=True, padding_value=0)
    
    # Encodec tokens (B, T_i) -> (B, T_max)
    encodec_tokens = pad_sequence(encodec_tokens, batch_first=True, padding_value=0)
    
    # Speaker embeddings
    speaker_embeddings = torch.stack(speaker_embeddings)
    
    return text_tokens, encodec_tokens, speaker_embeddings

def create_metavoice_dataloader(dataset_dir: str, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str, batch_size: int = 16, shuffle: bool = True):
    dataset = MetavoiceDataset(dataset_dir, encodec_model, tokenizer, spkemb_model, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)