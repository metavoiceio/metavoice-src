import os
import pathlib
import typing as tp

import julius
import torch
import torchaudio
from audiocraft.data.audio import audio_read
from encodec import EncodecModel
from torch.utils.data import Dataset

from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.fast_inference_utils import encode_tokens
from fam.llm.inference import SpeakerEncoder, TrainedBPETokeniser, get_cached_embedding
from fam.llm.utils import normalize_text

MBD_SAMPLE_RATE = 24000
END_OF_AUDIO_TOKEN = 1024

class MetavoiceData(Dataset):
    def __init__(self, dataset_dir: str, block_size: int, validation_split: float, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str, precision: torch.dtype):
        
        self.dataset_dir = dataset_dir
        self.block_size = block_size
        self.validation_split = validation_split
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.spkemb_model = spkemb_model
        self.device = device
        self.precision = precision

        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=END_OF_AUDIO_TOKEN)

        # Loop through dataset_dir and create a list of tuples (wav_path, text)
        # File system will look like:
        # dataset_dir/<utt_id>.wav and dataset_dir/<utt_id>.txt
        data_list = []
        for audio_file in pathlib.Path(dataset_dir).glob('*.wav'):
            utt_id = audio_file.stem
            wav_path = f"{dataset_dir}/{utt_id}.wav"
            txt_path = f"{dataset_dir}/{utt_id}.txt"
            with open(txt_path, 'r') as f:
                text = f.read()
            
            wav, sr = torchaudio.load(wav_path)
            if sr != MBD_SAMPLE_RATE:
                wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)
                torchaudio.save(wav_path, wav, MBD_SAMPLE_RATE)
            
            data_list.append((wav_path, text))
        
        self._prepare_dataset(data_list)
    
    def _prepare_dataset(self, data_list: tp.List[tp.Tuple[str, str]]):
        # We take data_list, extract all prompts and encodec tokens, and append them with EOT for all of them
        # This is done to prepare the dataset for the first stage of training

        full_sequence = torch.tensor([], dtype=torch.long, device=self.device)
        spk_embds = []
        current_wavs = torch.tensor([], dtype=torch.float, device=self.device)
        current_wav_duration = 0
        for wav_path, text in data_list:
            # Extract text tokenization
            prompt = self._extract_text_tokens(text)

            # Extract encodec tokens
            encodec_tokens = self._extract_encodec_tokens(wav_path)

            # Concatenate prompt and encodec tokens, and EOT token at the end
            eot = torch.tensor([END_OF_AUDIO_TOKEN], dtype=torch.long, device=self.device)
            sequence = torch.cat((prompt, encodec_tokens, eot))

            # Append to dataset
            # print("Encodec Tokens Length: ", encodec_tokens.size(0))
            # print("Prompt Length: ", prompt.size(0))
            # print("Tokenized Data Point length:", sequence.size(0))
            # print("Prompt: ", prompt)
            full_sequence = torch.cat((full_sequence, sequence), dim=-1)

            # Get wav data
            wav, sr = torchaudio.load(wav_path)  # Load the audio file
            if sr != MBD_SAMPLE_RATE:
                wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)
            if wav.ndim == 2:
                wav = wav.mean(dim=0)  # Average channels if stereo
            wav = wav.to(self.device)
            current_wavs = torch.cat((current_wavs, wav.unsqueeze(0)), dim=1)  # Concatenate along time axis
            current_wav_duration += wav.size(0) / MBD_SAMPLE_RATE
            if current_wav_duration >= 45: # 45 seconds
                current_wav_path = os.path.join(self.dataset_dir, "tmp_concatenated_wavs.wav")
                torchaudio.save(current_wav_path, current_wavs.cpu(), MBD_SAMPLE_RATE)
                
                # Extract speaker embeddings of the concatenated wav
                spk_emb = self._extract_speaker_embeddings(current_wav_path)
                spk_embds.append(spk_emb)
                
                # Reset
                current_wav_duration = 0
                current_wavs = torch.tensor([], dtype=torch.float32, device=self.device)
                os.remove(current_wav_path)
        
        # Split full_sequence into training and validation
        split = int(len(full_sequence) * (1 - self.validation_split))
        self.train_dataset = full_sequence[:split]
        self.val_dataset = full_sequence[split:]

        self.spk_embds = torch.stack(spk_embds) # (N, 1, 256)
    
    def get_batch(self, split: tp.Literal['train', 'val'], batch_size: int):
        if split == 'train':
            data = self.train_dataset
        elif split == 'val':
            data = self.val_dataset
        
        ix = torch.randint(0, data.size(0) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        # Random batch_size number of speaker embeddings
        spk_emb = self.spk_embds[torch.randint(0, self.spk_embds.size(0), (batch_size,))]

        return x, y, spk_emb
    
    def _extract_text_tokens(self, text: str):
        # For text tokens, one can use the tokenizer per:
        # https://github.com/metavoiceio/metavoice-src/blob/main/fam/llm/inference.py#L177
        text = normalize_text(text)
        encoded = encode_tokens(self.tokenizer, text, device=self.device)

        return encoded

    def _extract_encodec_tokens(self, wav_path: str):
        # read audio
        wav, sr = audio_read(wav_path)

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
        # Then add 1024 to 1st hierarchy tokens to match stage 1 output
        tokens = tokens.flatten().to(dtype=torch.int32) # (2*T)
        tokens[0::2] += END_OF_AUDIO_TOKEN
        
        return tokens

        # # Convert tokens to list before decoding to audio indices
        # tokens = tokens.tolist() # list[int]

        # # convert into audio ids
        # _, extracted_audio_ids = self.first_stage_adapter.decode([tokens])

        # # list[list[int], list[int]] -> (2, T), dtype long
        # encodec_tokens = torch.tensor(extracted_audio_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # # Interleave tokens and flatten (2, T) -> (2T,)
        # encodec_tokens = encodec_tokens.flatten() # (2T,)

        # return encodec_tokens # (2T,)

    def _extract_speaker_embeddings(self, wav_path: str):
        # For speaker embedding, you can also follow the code at:
        # https://github.com/metavoiceio/metavoice-src/blob/main/fam/llm/inference.py#L435
        return get_cached_embedding(wav_path, self.spkemb_model).to(self.device, dtype=self.precision)