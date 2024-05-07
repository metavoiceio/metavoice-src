"""
Module responsible for finetuning the first stage LLM.
"""

import itertools
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from fam.llm.config.finetune_params import *
from fam.llm.loaders.training_data import DynamicComputeDataset
from fam.llm.model import GPT, GPTConfig
from fam.llm.preprocessing.audio_token_mode import get_params_for_mode
from fam.llm.preprocessing.data_pipeline import get_training_tuple
from fam.llm.utils import hash_dictionary
from fam.telemetry import TelemetryEvent
from fam.telemetry.posthog import PosthogClient

# see fam/telemetry/README.md for more information
posthog = PosthogClient()

dtype: Literal["bfloat16", "float16", "tfloat32", "float32"] = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
seed_offset = 0

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True if dtype != "float32" else False
torch.backends.cudnn.allow_tf32 = True if dtype != "float32" else False
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "tfloat32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
    dtype
]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"tokens per iteration will be: {tokens_per_iter:,}")

ckpts_base_dir = pathlib.Path(__file__).resolve().parent / "ckpts"
if not os.path.exists(ckpts_base_dir) and master_process:
    print("Checkpoints directory didn't exist, creating...")
    ckpts_base_dir.mkdir(parents=True)

if master_process:
    if "/" in out_dir:
        raise Exception("out_dir should be just a name, not a path with slashes")

    ckpts_save_dir = ckpts_base_dir / out_dir
    os.makedirs(ckpts_save_dir, exist_ok=True)


def get_globals_state():
    """Return entirety of configuration global state which can be used for logging."""
    config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
    return {k: globals()[k] for k in config_keys}  # will be useful for logging


model_args: dict = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_sizes=None,
    dropout=dropout,
    causal=causal,
    norm_type=norm_type,
    rmsnorm_eps=rmsnorm_eps,
    nonlinearity_type=nonlinearity_type,
    spk_emb_on_text=spk_emb_on_text,
    attn_kernel_type=attn_kernel_type,
    swiglu_multiple_of=swiglu_multiple_of,
)  # start with model_args from command line


def strip_prefix(state_dict: Dict[str, Any], unwanted_prefix: str):
    # TODO: this also appears in fast_inference_utils._load_model, it should be moved to a common place.
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def force_ckpt_args(model_args, checkpoint_model_args) -> None:
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_sizes", "causal"]:
        model_args[k] = checkpoint_model_args[k]
    # this enables backward compatability with previously saved checkpoints.
    for k in [
        "target_vocab_sizes",
        "norm_type",
        "rmsnorm_eps",
        "nonlinearity_type",
        "attn_kernel_type",
        "spk_emb_on_text",
        "swiglu_multiple_of",
    ]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    if attn_kernel_type != model_args["attn_kernel_type"]:
        print(
            f'Found {model_args["attn_kernel_type"]} kernel type inside model,',
            f"but expected {attn_kernel_type}. Manually replacing it.",
        )
        model_args["attn_kernel_type"] = attn_kernel_type


@click.command()
@click.option("--train", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--val", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-id", type=str, required=False, default="metavoiceio/metavoice-1B-v0.1")
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path))
@click.option("--spk-emb-ckpt", type=click.Path(exists=True, path_type=Path))
def main(train: Path, val: Path, model_id: str, ckpt: Optional[Path], spk_emb_ckpt: Optional[Path]):
    if ckpt and spk_emb_ckpt:
        checkpoint_path, spk_emb_ckpt_path = ckpt, spk_emb_ckpt
    else:
        _model_dir = snapshot_download(repo_id=model_id)
        checkpoint_path = Path(f"{_model_dir}/first_stage.pt")
        spk_emb_ckpt_path = Path(f"{_model_dir}/speaker_encoder.pt")

    mode_params = get_params_for_mode(audio_token_mode, num_max_audio_tokens_timesteps=num_max_audio_tokens_timesteps)
    config = get_globals_state()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, map_location=device)
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", 1e9)
    checkpoint_model_args = checkpoint["model_args"]
    tokenizer_info = checkpoint.get("meta", {}).get("tokenizer", {})
    force_ckpt_args(model_args, checkpoint_model_args)
    gptconf = GPTConfig(**model_args)  # type: ignore
    model = GPT(gptconf, speaker_emb_dim=speaker_emb_size if speaker_cond else None)

    # removing torch.compile module prefixes for pre-compile loading
    state_dict = strip_prefix(checkpoint["model"], "_orig_mod.")
    model.load_state_dict(state_dict)
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if compile:
        print("Compiling the model... (takes a ~minute)")
        # requires PyTorch 2.0
        from einops._torch_specific import allow_ops_in_compiled_graph

        allow_ops_in_compiled_graph()
        model = torch.compile(model)  # type: ignore

    def estimate_loss(dataset, iters: int = eval_iters):
        """Estimate loss on a dataset by running on `iters` batches."""
        if dataset is None:
            return torch.nan
        losses = []
        for _, batch in zip(tqdm(range(iters)), dataset):
            X, Y, SE = get_training_tuple(batch, causal, num_codebooks, speaker_cond, device)
            with ctx:
                _, loss = model(X, Y, speaker_embs=SE, speaker_emb_mask=None)
            losses.append(loss.item())
        return torch.tensor(losses).mean()

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    if wandb_log and master_process:
        import wandb

        if os.environ.get("WANDB_RUN_ID", None) is not None:
            resume = "must"
        else:
            resume = None

        wandb.init(project=wandb_project, name=wandb_run_name, tags=wandb_tags, config=config, resume=resume)

    train_dataset = DynamicComputeDataset.from_meta(
        tokenizer_info,
        mode_params["combine_func"],
        spk_emb_ckpt_path,
        train,
        mode_params["pad_token"],
        mode_params["ctx_window"],
        device,
    )
    val_dataset = DynamicComputeDataset.from_meta(
        tokenizer_info,
        mode_params["combine_func"],
        spk_emb_ckpt_path,
        val,
        mode_params["pad_token"],
        mode_params["ctx_window"],
        device,
    )
    train_dataloader = itertools.cycle(DataLoader(train_dataset, batch_size, shuffle=True))
    train_data = iter(train_dataloader)
    # we do not perform any explicit checks for dataset overlap & leave it to the user
    # to handle this
    eval_val_data = DataLoader(val_dataset, batch_size, shuffle=True)
    # we can use the same Dataset object given it is a mapped dataset & not an iterable
    # one that can be exhausted. This implies we will be needlessly recomputing, fine
    # for now.
    eval_train_data = DataLoader(train_dataset, batch_size, shuffle=True)

    batch = next(train_data)
    X, Y, SE = get_training_tuple(batch, causal, num_codebooks, speaker_cond, device)

    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    total_norm = 0.0
    save_checkpoint = False
    if master_process:
        progress = tqdm(total=max_iters, desc="Training", initial=iter_num)
    else:
        progress = None

    # finetune last X transformer blocks and the ln_f layer
    trainable_count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Before layer freezing {trainable_count(model)=}...")
    for param in model.parameters():
        param.requires_grad = False
    for param in itertools.chain(
        model.transformer.ln_f.parameters(), model.transformer.h[last_n_blocks_to_finetune * -1 :].parameters()
    ):
        param.requires_grad = True
    print(f"After freezing excl. last {last_n_blocks_to_finetune} transformer blocks: {trainable_count(model)=}...")

    # log start of finetuning event
    properties = {
        **config,
        **model_args,
        "train": str(train),
        "val": str(val),
        "model_id": model_id,
        "ckpt": ckpt,
        "spk_emb_ckpt": spk_emb_ckpt,
    }
    finetune_jobid = hash_dictionary(properties)
    posthog.capture(
        TelemetryEvent(
            name="user_started_finetuning",
            properties={"finetune_jobid": finetune_jobid, **properties},
        )
    )

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if master_process:
            if iter_num % eval_interval == 0 and master_process:
                ckpt_save_name = f"ckpt_{iter_num:07d}.pt"
                with torch.no_grad():
                    model.eval()
                    losses = {
                        "train": estimate_loss(eval_train_data),
                        "val": estimate_loss(eval_val_data),
                    }
                    model.train()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                            "stats/total_norm": total_norm,
                        }
                    )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        ckpt_save_name = ckpt_save_name.replace(
                            ".pt", f"_bestval_{best_val_loss}".replace(".", "_") + ".pt"
                        )
                        save_checkpoint = True

                save_checkpoint = save_checkpoint or iter_num % save_interval == 0
                if save_checkpoint and iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),  # type: ignore
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "meta": {
                            "speaker_cond": speaker_cond,
                            "speaker_emb_size": speaker_emb_size,
                            "tokenizer": tokenizer_info,
                        },
                    }
                    torch.save(checkpoint, os.path.join(ckpts_save_dir, ckpt_save_name))
                    print(f"saving checkpoint to {ckpts_save_dir}")
                    save_checkpoint = False
            if iter_num == 0 and eval_only:
                break
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1  # type: ignore
                with ctx:  # type: ignore
                    logits, loss = model(X, Y, speaker_embs=SE, speaker_emb_mask=None)
                    loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                batch = next(train_data)
                X, Y, SE = get_training_tuple(
                    batch,
                    causal,
                    num_codebooks,
                    speaker_cond,
                    device,
                )
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                progress.update(1)
                progress.set_description(f"Training: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                if iter_num % log_interval == 0:
                    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                # log end of finetuning event
                posthog.capture(
                    TelemetryEvent(
                        name="user_completed_finetuning",
                        properties={"finetune_jobid": finetune_jobid, "loss": round(lossf, 4)},
                    )
                )
                break


if __name__ == "__main__":
    main()
