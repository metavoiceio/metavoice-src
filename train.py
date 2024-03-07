import os
from pathlib import Path

import torch
import tqdm
from encodec import EncodecModel
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import SGD, AdamW
import typing as tp

from dataloader import create_metavoice_dataloader
from fam.llm.fast_inference_utils import (
    build_model,
    _load_model,
    device_sync,
    logits_to_probs
)
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.utils import get_default_dtype, get_device
from lora import TransformerWithLoRA, freeze_parameters_except_lora

from accelerate import Accelerator

def _get_batch(device, precision):
    # Generate fake batch
    # With prompts, encodec_tokens, speaker_embeddings
    # B = 2
    # P = 1000
    # V = 2562
    # T = 1000
    # prompts(B, P) --> long
    # encodec_tokens(B, T) --> int
    # speaker_embeddings(B, 1, 256) --> float
    # input_pos(T) --> int (0 to P-1)
    # targets(B, T) --> int
    prompts = torch.randint(0, 2562, (2, 1000), device=device, dtype=torch.long)
    encodec_tokens = torch.randint(0, 2562, (2, 1000), device=device, dtype=torch.long)
    speaker_embeddings = torch.randn(2, 1, 256, device=device, dtype=precision)
    input_pos = torch.arange(1000, device=device, dtype=torch.long)
    targets = torch.randint(0, 2562, (2, 1000), device=device, dtype=torch.long)

    return prompts, encodec_tokens, speaker_embeddings, input_pos, targets


# Launch this (train.py) with:
# FOR bfloat16:     "accelerate launch --mixed_precision=bf16 train.py"
# FOR float16:      "accelerate launch --mixed_precision=f16 train.py"

accelerator = Accelerator()

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
        # self._device = get_device()
        self._device = str(accelerator.device)
        self._model_dir = 'models'
        # self._model_dir = snapshot_download(repo_id=model_name)

        self.encodec_model = EncodecModel.encodec_model_24khz().to(self._device)
        self.encodec_model.set_target_bandwidth(self.ENCODEC_BANDWIDTH)

        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=self.END_OF_AUDIO_TOKEN)

        print("building model")
        self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        # self.model, self.tokenizer, self.smodel, self.model_size = build_model(
        #     precision=self.precision,
        #     checkpoint_path=Path(f"{self._model_dir}/first_stage.pt"),
        #     spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
        #     device=self._device,
        #     compile=False,
        #     compile_prefill=True,
        # )

        self.model, self.tokenizer, self.smodel = _load_model(
            checkpoint_path=Path(f"{self._model_dir}/first_stage.pt"),
            spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
            device=self._device,
            precision=self.precision,
        )
        device_sync(device=self._device)  # MKG
        with torch.device(self._device):
            self.model.setup_spk_cond_mask()
            self.model.setup_caches(
                max_batch_size=2,
                max_seq_length=self.model.config.block_size,
            )
        device_sync(device=self._device)  # MKG

        # Add LoRA to stage 1 model for finetuning
        self.model = TransformerWithLoRA.from_base_model(self.model, device=self._device, precision=self.precision)

    def train(self, training_name: str, epochs=100, learning_rate=0.0001):
        print("Initializing dataloader...")

        # Hyperparameters
        # TODO(hyperparameters) add hyperparameters as parameters in function
        batch_size = 2

        # Model configuration for training
        top_p = 0.95
        guidance_scale = 3.0
        temperature = 1.0

        temperature = torch.tensor(temperature, device=self._device, dtype=self.precision)
        top_k = None
        top_p = torch.tensor(top_p, device=self._device, dtype=self.precision)
        guidance_scale = torch.tensor(guidance_scale, device=self._device, dtype=self.precision)

        dataloader = create_metavoice_dataloader(
            dataset_dir=self._dataset_dir,
            encodec_model=self.encodec_model,
            tokenizer=self.tokenizer, 
            spkemb_model=self.smodel, 
            batch_size=2, 
            device=self._device
        )
        # TODO(validation data) load validation data as well for val loss

        # Freeze stage 1 model parameters except LoRA layers
        print("Freezing model parameters (except LoRA layers)...")
        freeze_parameters_except_lora(self.model)

        print("Initializing optimizer and scaler...")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scaler = GradScaler()

        # Prepare model, optimizer, and dataloader for mixed precision training
        print("Preparing model, optimizer, and dataloader for mixed precision training...")
        self.model, optimizer, dataloader = accelerator.prepare(self.model, optimizer, dataloader)
        
        # Put model in training mode
        print("Setting model to training mode...")
        self.model.train()

        # Ensure the directory for saving models exists
        print("Preparing model folder...")
        save_dir = os.path.join('saved_models', training_name)
        os.makedirs(save_dir, exist_ok=True)

        print("Number of batches: ", len(dataloader))
        print("Starting training...")
        test_printed = False
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (prompts, encodec_tokens, speaker_embeddings) in enumerate(dataloader):
                # B = batch size
                # P = length of prompt
                # V = vocab size (2562 for metavoice-1b)
                # T = length of encodec tokens in batch

                # prompts -> (B, P)
                # encodec_tokens -> (B, T) <flattened interleaved 2 hierarchies>

                B = prompts.shape[0]
                P = prompts.shape[1]
                V = self.model.config.vocab_size
                T = encodec_tokens.shape[-1]

                if B != batch_size:
                    # This would be the case for the last batch
                    # if len(dataloader) % batch_size != 0
                    # eg if batch size = 2 and len(dataloader) is uneven
                    # then the last batch will be 1, causing issues.

                    # When B != batch_size, it breaks KVCache in the model, 
                    # because KVCache can't adapt to dynamic batch sizes.
                    # For now we just skip the last batch to fix.
                    continue

                # 1. Determine random encodec token location
                # [0, T_rand-1]
                T_rand = torch.randint(0, T, (1,)).item()
                S = T_rand + P

                # 2. Compute input_pos
                # input_pos -> (T)
                input_pos = torch.arange(S, device=self._device, dtype=torch.long)

                # 3. Arrange input data
                # X -> (B, P + T_rand)
                X = torch.cat([prompts, encodec_tokens[:, :T_rand]], dim=1)

                # 4. Arrange target data
                # Y -> (B, P + T_rand) <X but shifted by 1 to the right>
                Y = torch.cat([prompts, encodec_tokens[:, 1:T_rand+1]], dim=1)

                if not test_printed:
                    print("Y shape: ", Y.shape)
                    print("Y min value: ", Y.min())
                    print("Y max value: ", Y.max())
                    print("X shape: ", X.shape)
                    print("speaker_embeddings shape: ", speaker_embeddings.shape)
                    print("input_pos shape: ", input_pos.shape)
                    print("Y shape: ", Y.shape)
                    test_printed = True

                with autocast(dtype=self.precision):
                    # 5. Compute loss
                    optimizer.zero_grad()
                    _, loss = self.model(
                        X, 
                        speaker_embeddings, 
                        input_pos,
                        targets=Y,
                        debug_mode=True,
                    )

                # 6. Backward pass
                accelerator.backward(loss)

                # 7. Step optimizer
                optimizer.step()

                # 8. Update total loss
                total_loss += loss.item()
                
                # 9. Clear cache in model to prevent persistent tensor with gradients between iterations
                self.model.clear_and_detach_caches()
                
                # Print batch loss
                print(f'Batch {batch_idx+1}, Loss: {loss.item():.4f}\n')
            
            # TODO(validation loss) implement validation loss

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')
            
            # Save model after each epoch
            # epoch_save_path = os.path.join(save_dir, f'{training_name}_epoch_{epoch+1}.pt')
            # torch.save(self.model.state_dict(), epoch_save_path)
            # print(f'Model saved to {epoch_save_path}')

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
    trainer = MetaVoiceTrainer(
        model_dir=model_dir,
        dataset_dir=dataset_dir
    )

    print("Training...")
    # trainer.train_test()
    trainer.train(
        training_name
    )