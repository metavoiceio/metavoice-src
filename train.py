import os
from pathlib import Path

import torch
import tqdm
from encodec import EncodecModel
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
import typing as tp

from dataloader import create_metavoice_dataloaders
from fam.llm.fast_inference_utils import (
    build_model,
    _load_model,
    device_sync,
    logits_to_probs
)
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.utils import get_default_dtype, get_device
from lora import TransformerWithLoRA, get_lora_model

from accelerate import Accelerator

# Launch this (train.py) with:
# FOR bfloat16:     "accelerate launch --mixed_precision=bf16 train.py"
# FOR float16:      "accelerate launch --mixed_precision=f16 train.py"

accelerator = Accelerator()

class MetaVoiceTrainer:
    ENCODEC_BANDWIDTH = 6.0
    END_OF_AUDIO_TOKEN = 1024

    def __init__(self, model_dir: str = 'models', dataset_dir: str = 'dataset'):
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

        # Add LoRA to model for finetuning
        self.model = TransformerWithLoRA(self.model)

    def train(self, training_name: str, epochs=100, learning_rate=2e-5):
        print("Initializing dataloader...")

        # Hyperparameters
        # TODO(hyperparameters) add hyperparameters as parameters in function
        batch_size = 2
        validation_split = 0.2
        shuffle = True
        save_epochs = epochs + 1 # Just disable for now
        log_epochs = 10
        eval_epochs = 10
        weight_decay = 0.001

        train_dataloader, validation_dataloader = create_metavoice_dataloaders(
            dataset_dir=self._dataset_dir,
            encodec_model=self.encodec_model,
            tokenizer=self.tokenizer,
            spkemb_model=self.smodel,
            device=self._device,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle,
        )

        print("Initializing optimizer and scaler...")
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Prepare model, optimizer, and dataloader for mixed precision training
        print("Preparing model, optimizer, and dataloader for mixed precision training...")
        self.model, optimizer, train_dataloader, validation_dataloader = accelerator.prepare(self.model, optimizer, train_dataloader, validation_dataloader)

        # Put model in training mode
        print("Setting model to training mode...")
        self.model.train()

        # Ensure the directory for saving models exists
        print("Preparing model folder...")
        save_dir = os.path.join('saved_models', training_name)
        os.makedirs(save_dir, exist_ok=True)

        print("Number of training batches: ", len(train_dataloader))
        print("Starting training...")
        test_printed = False
        nan_printed = False
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (X, Y, spk_embds, input_pos) in enumerate(train_dataloader):
                B = X.shape[0]
                if B != batch_size:
                    # Batch size is not consistent, skip this batch.
                    # This is a temporary fix, root cause should be fixed instead.
                    # Error is caused in KVCache, where the batch size cannot be dynamic
                    # as it is set at the initialization of the model.
                    continue
                
                if not test_printed:
                    print("Y shape: ", Y.shape)
                    print("Y min value: ", Y.min())
                    print("Y max value: ", Y.max())
                    print("X shape: ", X.shape)
                    print("spk_embds shape: ", spk_embds.shape)
                    print("input_pos shape: ", input_pos.shape)
                    print("Y shape: ", Y.shape)
                    test_printed = True

                with autocast(dtype=self.precision):
                    # 5. Compute loss
                    optimizer.zero_grad()
                    logits, loss = self.model(
                        X, 
                        spk_embds, 
                        input_pos,
                        targets=Y,
                        debug_mode=False,
                    )

                # 6. Backward pass
                accelerator.backward(loss)

                # 7. Step optimizer
                optimizer.step()

                # 8. Update total loss
                print("loss")
                print(loss)
                total_loss += loss.item()

                # check if loss is nan
                if nan_printed == False and torch.isnan(loss).any():
                    print("loss is nan!") 
                    print("batch_idx: ", batch_idx)
                    
                    print("logits: ", logits)
                    print("logits shape: ", logits.shape, "\n\n")

                    print("X: ", X)
                    print("X shape: ", X.shape, "\n\n")
                    
                    print("Y: ", Y)
                    print("Y shape: ", Y.shape, "\n\n")

                    print("input_pos: ", input_pos)
                    print("input_pos shape: ", input_pos.shape, "\n\n")

                    print("spk_embds: ", spk_embds)
                    print("spk_embds shape: ", spk_embds.shape, "\n")

                    nan_printed = True
                
                # 9. Clear cache in model to prevent persistent tensor with gradients between iterations
                self.model.clear_and_detach_caches()
            
            print("len train_dataloader")
            print(len(train_dataloader))
            print("total loss")
            print(total_loss)

            avg_loss = total_loss / len(train_dataloader)

            # Print average loss every log_epochs
            if (epoch + 1) % log_epochs == 0:
                print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')
            
            # Save model every save_epochs
            if (epoch + 1) % save_epochs == 0:
                print(f'Saving model at epoch {epoch+1}...')
                epoch_save_path = os.path.join(save_dir, f'{training_name}_epoch_{epoch+1}.pt')
                torch.save(self.model.state_dict(), epoch_save_path)
                print(f'Model saved to {epoch_save_path}')
            
            if (epoch + 1) % eval_epochs == 0:
                # Evaluate model
                print("Evaluating model...")
                losses = self.estimate_loss(validation_dataloader)
                train_loss = losses['train']
                val_loss = losses['val']
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        print("Training complete")
    
    @torch.no_grad()
    def estimate_loss(self, val_dataloader: DataLoader, eval_iters: int = 50):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in tqdm.tqdm(range(eval_iters), desc=f'Estimating {split} loss'):
                X, Y, spk_embds, input_pos = next(iter(val_dataloader))
                with autocast():
                    _, loss = self.model(
                        X, 
                        spk_embds, 
                        input_pos,
                        targets=Y,
                        debug_mode=False,
                    )
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

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