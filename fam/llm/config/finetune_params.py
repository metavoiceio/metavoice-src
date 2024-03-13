from contextlib import nullcontext
import os
import uuid
import pathlib
from typing import Literal, Optional
import torch

batch_size = 2
dataset_size: int = 400
batched_ds_size = dataset_size // batch_size
val_train_ratio = 0.2

epochs: int = 2
max_iters = batched_ds_size * epochs
learning_rate = 3e-5
last_n_blocks_to_finetune = 1
decay_lr = False
lr_decay_iters = 0 # decay learning rate after this many iterations
min_lr = 3e-6

eval_interval = batched_ds_size
eval_iters = int(batched_ds_size*val_train_ratio)
eval_only: bool = False # if True, script exits right after the first eval
log_interval = batched_ds_size # don't print too too often
save_interval: int = batched_ds_size * (epochs//2) # save a checkpoint every this many iterations
assert save_interval % eval_interval == 0, "save_interval must be divisible by eval_interval."
seed = 1337
grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

wandb_log = False
wandb_project = "project-name"
wandb_run_name = "run-name"
wandb_tags = ["tag1", "tag2"]

gradient_accumulation_steps = 1
block_size = 2_048
audio_token_mode = "flattened_interleaved"
num_max_audio_tokens_timesteps = 1_024

n_layer = 24
n_head = 16
n_embd = 2048
dropout = 0.1

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

warmup_iters: int = 0 # how many steps to warm up for
out_dir = f"finetune-{epochs=}-{learning_rate=}-{batch_size=}-{last_n_blocks_to_finetune=}-{dropout=}-{uuid.uuid4()}"

compile = True
num_codebooks = None
norm_type = "rmsnorm"
rmsnorm_eps = 1e-5
nonlinearity_type = "swiglu"
swiglu_multiple_of = 256
attn_kernel_type = "torch_attn"
meta_target_vocab_sizes: Optional[list[int]] = None
speaker_emb_size: int = 256
speaker_cond = True

# always running finetuning on a single GPU
master_process = True
device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
ddp = False
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

causal = True
bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
spk_emb_on_text: bool = True  # whether to add speaker embedding conditioning to text tokens or not
