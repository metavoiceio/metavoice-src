import os
from pathlib import Path

import torch
import tqdm
from accelerate import Accelerator
from encodec import EncodecModel
from torch.cuda.amp import autocast
from torch.optim import AdamW

from dataloader import MetavoiceData
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.fast_inference_utils import _load_model, device_sync
from fam.llm.utils import get_default_dtype
from lora import TransformerWithLoRA

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
                use_kv_cache=False,
            )
        device_sync(device=self._device)  # MKG
        
        # Add LoRA to model for finetuning
        self.model = TransformerWithLoRA(self.model)
        self.model = self.model.to(self._device)

    def train(self, training_name: str, epochs=100, learning_rate=1e-4):
        # Hyperparameters
        # TODO(hyperparameters) add hyperparameters as parameters in function
        batch_size = 2
        validation_split = 0.1
        save_epochs = 5 # Just disable for now
        log_epochs = 5
        eval_epochs = 20
        weight_decay = 0.001
        beta1 = 0.9
        beta2 = 0.95
        grad_clip = 1.0
        gradient_accumulation_steps = 5 * 2
        block_size = self.model.config.block_size # 2048 for metavoice-1b

        print("Initializing dataloader...")
        data = MetavoiceData(
            dataset_dir=self._dataset_dir,
            block_size=block_size,
            validation_split=validation_split,
            encodec_model=self.encodec_model,
            tokenizer=self.tokenizer,
            spkemb_model=self.smodel,
            device=self._device,
        )

        print("Initializing optimizer and scaler...")
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )

        # Prepare model, optimizer, and dataloader for mixed precision training
        print("Preparing model, optimizer, and dataloader for mixed precision training...")
        self.model, optimizer = accelerator.prepare(self.model, optimizer)
        
        # Put model in training mode
        print("Setting model to training mode...")
        self.model.train()

        # Ensure the directory for saving models exists
        print("Preparing model folder...")
        save_dir = os.path.join('saved_models', training_name)
        os.makedirs(save_dir, exist_ok=True)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model has {n_params} parameters, {n_trainable_params} of which are trainable")

        print("Starting training...")
        test_printed = False
        for epoch in range(epochs):
            for micro_step in range(gradient_accumulation_steps):
                X, Y, spk_embds = data.get_batch('train', batch_size)
                input_pos = torch.arange(block_size, device=self._device)
                
                if not test_printed:
                    print(X[0])
                    print("^^^^X[0]^^^^")
                    
                    print(Y[0])
                    print("^^^^Y[0]^^^^")
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
                
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # 6. Backward pass
                accelerator.backward(loss)

            # 7. Clip the gradient
            if grad_clip != 0.0:
                accelerator.unscale_gradients(optimizer)
                accelerator.clip_grad_norm_(self.model.base_model.parameters(), grad_clip)

            # 8. Step optimizer
            optimizer.step()

            # Print average loss every log_epochs
            if (epoch + 1) % log_epochs == 0:
                lossf = loss.item() * gradient_accumulation_steps
                print(f'Epoch {epoch+1}, Loss: {lossf:.4f}')
            
            # Save model every save_epochs
            if (epoch + 1) % save_epochs == 0:
                print(f'Saving LoRA at epoch {epoch+1}...')
                lora_epoch_save_path = os.path.join(save_dir, f'lora_epoch_{epoch+1}.pt')
                self.model.save_lora(lora_epoch_save_path)
                print(f'Model saved to {lora_epoch_save_path}!')
            
            if (epoch + 1) % eval_epochs == 0:
                # Evaluate model
                print("Evaluating model...")
                losses = self.estimate_loss(data)
                train_loss = losses['train']
                val_loss = losses['val']
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        print("Training complete")
    
    @torch.no_grad()
    def estimate_loss(self, data: MetavoiceData, eval_iters: int = 50):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in tqdm.tqdm(range(eval_iters), desc=f'Estimating {split} loss'):
                X, Y, spk_embds = data.get_batch(split, 2)
                input_pos = torch.arange(self.model.config.block_size, device=self._device)
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
    dataset_dir = 'daniel_dataset'
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