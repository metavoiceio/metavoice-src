import os
from pathlib import Path

import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.autograd import Variable
import tempfile
import shutil
import tqdm
import librosa

from typing import Optional

from encodec import EncodecModel

from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.decoders import EncodecDecoder
from fam.llm.fast_inference_utils import build_model, main, prefill, decode_one_token, decode_n_tokens, sample, model_forward
from fam.llm.inference import (
    EncodecDecoder,
    InferenceConfig,
    Model,
    TiltedEncodec,
    TrainedBPETokeniser,
    get_enhancer,
)
from fam.llm.utils import (
    get_default_dtype,
    get_device,
    normalize_text,
)

from lora import TransformerWithLoRA, freeze_parameters_except_lora

from dataloader import create_metavoice_dataloader

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


class MetaVoiceTrainer:
    ENCODEC_BANDWIDTH = 6.0
    END_OF_AUDIO_TOKEN = 1024

    def __init__(self, model_dir: str = 'models', dataset_dir: str = 'dataset', train_stage_two: bool = False, seed: int = 1337):
        self._model_dir = model_dir
        self._dataset_dir = dataset_dir
        self._output_dir = f"{self._model_dir}/outputs"

        os.makedirs(self._output_dir, exist_ok=True)

        # NOTE: this needs to come first so that we don't change global state when we want to use
        # the torch.compiled-model.
        self._dtype = get_default_dtype()
        self._device = get_device()
        self._model_dir = 'models'
        # self._model_dir = snapshot_download(repo_id=model_name)

        self.encodec_model = EncodecModel.encodec_model_24khz().to(self._device)
        self.encodec_model.set_target_bandwidth(self.ENCODEC_BANDWIDTH)

        print("Models...")
        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=self.END_OF_AUDIO_TOKEN)

        second_stage_ckpt_path = f"{self._model_dir}/second_stage.pt"
        config_second_stage = InferenceConfig(
            ckpt_path=second_stage_ckpt_path,
            num_samples=1,
            seed=seed,
            device=self._device,
            dtype=self._dtype,
            compile=False,
            init_from="resume",
            output_dir=self._output_dir,
        )
        data_adapter_second_stage = TiltedEncodec(end_of_audio_token=self.END_OF_AUDIO_TOKEN)
        self.llm_second_stage = Model(
            config_second_stage, TrainedBPETokeniser, EncodecDecoder, data_adapter_fn=data_adapter_second_stage.decode
        )
        self.enhancer = get_enhancer("df")

        print("building model")
        self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        self.model, self.tokenizer, self.smodel, self.model_size = build_model(
            precision=self.precision,
            checkpoint_path=Path(f"{self._model_dir}/first_stage.pt"),
            spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
            device=self._device,
            compile=False,
            compile_prefill=True,
        )

        # Add LoRA to stage 1 model for finetuning
        self.model = TransformerWithLoRA.from_base_model(self.model)

    def _train_infer_stage_one(
            self,
            prompt: torch.Tensor,
            spk_emb: torch.Tensor,
            guidance_scale: torch.Tensor,
            temperature: torch.Tensor,
            top_p: Optional[torch.Tensor] = None,
            top_k: Optional[torch.Tensor] = None,
            max_new_tokens: Optional[int] = None,
            end_of_audio_token: int = 2048,
            device: str = "cuda",
    ) -> torch.Tensor:
        # Should return decoded Encodec tokens from stage 1 LLM as a torch.Tensor
        T = prompt.size(0)
        if max_new_tokens is None:
                max_seq_length = self.model.config.block_size
        else:
            max_seq_length = T + max_new_tokens
            max_seq_length = min(max_seq_length, self.model.config.block_size)
        max_new_tokens = max_seq_length - T
        if max_new_tokens <= 0:
            raise ValueError("Prompt is too long to generate more tokens")
    
        device, dtype = prompt.device, prompt.dtype

        seq = torch.clone(prompt)
        input_pos = torch.arange(0, T, device=device)

        next_token: torch.Tensor = prefill(
            self.model, 
            prompt.view(1, -1).repeat(2, 1), 
            spk_emb, 
            input_pos, 
            guidance_scale = guidance_scale, 
            temperature = temperature, 
            top_p = top_p, 
            top_k = top_k
        )
        seq = torch.cat([seq, next_token.view(1)])

        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        generated_tokens, _ = decode_n_tokens(
            self.model,
            next_token.view(1, -1).repeat(2, 1),
            spk_emb,
            input_pos,
            max_new_tokens - 1,
            callback=lambda x: x,
            end_of_audio_token=end_of_audio_token,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        seq = torch.cat([seq, torch.cat(generated_tokens)])

        tokens = seq.tolist()

        _, extracted_audio_ids = self.first_stage_adapter.decode([tokens])

        return [torch.tensor(extracted_audio_ids, dtype=torch.int32, device=self._device).unsqueeze(0)]

    def train(self, training_name: str, epochs=10, learning_rate=1e-4):
        print("Initializing dataloader...")
        print("SECOND STAGE CAUSAL:")
        print(self.llm_second_stage.model.config.causal)

        # Model configuration for training
        top_p = 0.95
        guidance_scale = 3.0
        temperature = 1.0

        dataloader = create_metavoice_dataloader(
            dataset_dir=self._dataset_dir,
            encodec_model=self.encodec_model,
            tokenizer=self.tokenizer, 
            spkemb_model=self.smodel, 
            batch_size=2, 
            device=self._device)
        # TODO(validation data) load validation data as well for val loss

        print("Setting model to training mode...")
        self.model.train()  # Set the model to training mode
        self.llm_second_stage.model.train()  # Set the model to training mode

        # Freeze stage 1 model parameters except LoRA layers
        freeze_parameters_except_lora(self.model)

        print("Initializing optimizer and scaler...")
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scaler = GradScaler()

        # Ensure the directory for saving models exists
        print("Preparing model folder...")
        save_dir = os.path.join('saved_models', training_name)
        os.makedirs(save_dir, exist_ok=True)

        print("Starting training...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (text_tokens, wavs, speaker_embeddings, texts) in enumerate(dataloader):
                print("Batch ", (batch_idx + 1))

                optimizer.zero_grad()
                
                batch_loss = 0.0  # Initialize batch loss
                for prompt, wav_gt, speaker_embedding, text in zip(text_tokens, wavs, speaker_embeddings, texts):
                    prompt = prompt.to(self._device)
                    wav_gt = wav_gt.to(self._device)
                    speaker_embedding = speaker_embedding.to(self._device)

                    with autocast():
                        # Stage 1:
                        # Text ---> Encodec tokens x2 hierarchies
                        encodec_tokens = self._train_infer_stage_one(
                            prompt=prompt,
                            spk_emb=speaker_embedding,
                            guidance_scale=torch.tensor(guidance_scale, device=self._device, dtype=self.precision),
                            temperature=torch.tensor(temperature, device=self._device, dtype=self.precision),
                            top_p=torch.tensor(top_p, device=self._device, dtype=self.precision),
                            top_k=None,
                            max_new_tokens=None,
                            end_of_audio_token=None,
                            device=self._device
                        )

                    b_speaker_embs = speaker_embedding.unsqueeze(0)

                    # Stage 2:
                    # Encodec tokens x2 hierarchies ---> Waveform
                    # TODO(training) Runs fine, but doesn't support training yet.
                    with torch.no_grad():
                        wav_files = self.llm_second_stage(
                            texts=[text],
                            encodec_tokens=encodec_tokens,
                            speaker_embs=b_speaker_embs,
                            batch_size=1,
                            guidance_scale=None,
                            top_p=None,
                            top_k=200,
                            temperature=1.0,
                            max_new_tokens=None
                        )

                        print("Second stage LLM output")
                        print(wav_files)

                        # enhance using deepfilternet
                        wav_file = wav_files[0]
                        with tempfile.NamedTemporaryFile(suffix=".wav") as enhanced_tmp:
                            self.enhancer(str(wav_file) + ".wav", enhanced_tmp.name)
                            shutil.copy2(enhanced_tmp.name, str(wav_file) + ".wav")
                            print(f"\nSaved audio to {wav_file}.wav")
                    
                    # Get wav file for calculating loss function
                    wav, _ = librosa.load(str(wav_file) + ".wav")
                    wav = torch.tensor(wav, dtype=self.precision, device=self._device)
                    wav = wav.unsqueeze(0)
                    wav = wav.unsqueeze(0)
                    wav = wav.to(self._device, dtype=self.precision)

                    print("wav shape: ", wav.shape)
                    print("wav_gt shape: ", wav_gt.shape)
                    
                    # Ensure wav and wav_gt have shapes ready for loss function
                    length_wav, length_wav_gt = wav.size(-1), wav_gt.size(-1)
                    max_length = max(length_wav, length_wav_gt)
                    
                    # Pad wav and wav_gt to the same length (max_length)
                    wav = F.pad(wav, (0, max_length - length_wav))
                    wav_gt = F.pad(wav_gt, (0, max_length - length_wav_gt))

                    # Compute loss between wav and wav_gt
                    loss = F.mse_loss(wav, wav_gt)
                    batch_loss += loss.item()
                    total_loss += loss.item()

                    # Backpropagation with gradient scaling for mixed precision training
                    scaler.scale(loss).backward()
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Clear gradients at the end of the batch processing

                # Print batch loss
                print(f'Batch {batch_idx+1}, Loss: {loss.item():.4f}')
            
            # TODO(validation loss) implement validation loss

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')
            
            # Save model after each epoch
            epoch_save_path = os.path.join(save_dir, f'{training_name}_epoch_{epoch+1}.pt')
            torch.save(self.model.state_dict(), epoch_save_path)
            print(f'Model saved to {epoch_save_path}')

        print("Training complete")

if __name__ == '__main__':
    model_dir = 'models'

    # Expected data directory format:
    # <dataset_dir>/<UTT_1>.wav <dataset_dir>/<UTT_1>.txt
    # <dataset_dir>/<UTT_2>.wav <dataset_dir>/<UTT_2>.txt
    # <dataset_dir>/<UTT_3>.wav <dataset_dir>/<UTT_3>.txt
    # txt files should contain the transcript of the audio
    # etc...
    dataset_dir = 'dummy_dataset'
    training_name = 'finetune_001'


    print("Initializing trainer...")
    trainer = MetaVoiceTrainer()

    print("Training...")
    trainer.train(
        training_name
    )