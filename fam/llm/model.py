import inspect
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import tqdm
from einops import rearrange
from torch.nn import functional as F

from fam.llm.layers import Block, LayerNorm, RMSNorm
from fam.llm.mixins import CausalInferenceMixin, NonCausalInferenceMixin

END_OF_TEXT_TOKEN = 1537


def _select_spkemb(spkemb, mask):
    _, examples, _ = spkemb.shape
    mask = torch.nn.functional.one_hot(mask.long(), num_classes=examples).to(spkemb)  # shape: (batch, time, examples)
    spkemb = spkemb.transpose(1, 2)  # b ex c -> b c ex
    mask = mask.transpose(1, 2)  # b t ex -> b ex t
    return torch.bmm(spkemb, mask).transpose(1, 2)  # b c t -> b t c


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_sizes: list = field(default_factory=list)
    target_vocab_sizes: Optional[list] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    spkemb_dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = (
        True  # auto-regressive or not, i.e. whether to have attention mask that prevents attending to future tokens
    )
    spk_emb_on_text: bool = True  # whether to add speaker embedding conditioning to text tokens or not
    norm_type: str = "layernorm"  # "rmsnorm" or "layernorm
    rmsnorm_eps: Optional[float] = None  # only used for rmsnorm
    nonlinearity_type: str = "gelu"  # "gelu" or "swiglu"
    swiglu_multiple_of: Optional[int] = None  # MLP hidden layer (using SwiGLU) will be multiple of this
    attn_kernel_type: Literal["fd", "torch_attn"] = "torch_attn"
    kv_cache_enabled: bool = False  # whether to use key-value cache for attention


def _check_speaker_emb_dims(
    speaker_embs: Union[list, torch.Tensor], expected_speaker_emb_dim: int, expected_batch_size: int
) -> Union[torch.Tensor, list]:
    """
    Checks that the speaker embedding dimensions are correct, and reshapes them if necessary.
    """
    if type(speaker_embs) == list:
        b_se = len(speaker_embs)
        for i, s in enumerate(speaker_embs):
            if s is not None:
                emb_dim = s.shape[-1]
                if s.ndim == 1:
                    speaker_embs[i] = speaker_embs[i].unsqueeze(0)
    else:
        if speaker_embs.ndim == 2:
            # if we have a single speaker embedding for the whole sequence,
            # add a dummy dimension for backwards compatibility
            speaker_embs = speaker_embs[:, None, :]

        # num_examples is the number of utterances packed into this sequence
        b_se, num_examples, emb_dim = speaker_embs.size()

    assert b_se == expected_batch_size, f"Batch size mismatch: {b_se} != {expected_batch_size}"
    assert (
        emb_dim == expected_speaker_emb_dim
    ), f"Speaker embedding dimension mismatch: {emb_dim} != {expected_speaker_emb_dim}"

    return speaker_embs


class GPT(nn.Module, NonCausalInferenceMixin, CausalInferenceMixin):
    def __init__(self, config: GPTConfig, speaker_emb_dim: Optional[int] = None):
        """
        Initialize the GPT model.

        Args:
            config (GPTConfig): Configuration object for the model.
            speaker_emb_dim (Optional[int]): Dimension of the speaker embedding. Default is None.
        """
        super().__init__()
        assert config.vocab_sizes is not None
        assert config.block_size is not None
        self.config = config

        self.kv_cache_enabled = False  # disabled by default
        self.kv_pos = 0

        self.speaker_emb_dim = speaker_emb_dim
        self.spk_emb_on_text = config.spk_emb_on_text
        if self.config.causal is True and self.spk_emb_on_text is False:
            print("!!!!!!!!!!!!!!!!!!")
            print(
                f"!!!!!!!! Using DEFAULT of {END_OF_TEXT_TOKEN} as end of text token to find speaker cond masking!! You likely need to change this."
            )
            print("!!!!!!!!!!!!!!!!!!")
        if self.config.causal is False and self.spk_emb_on_text is False:
            raise Exception(
                "Cannot use speaker embedding masking with non-causal model. This is unexpected. Check for relevant changes required in code before proceeding."
            )

        if config.norm_type == "rmsnorm":
            if config.rmsnorm_eps is None:
                raise Exception("RMSNorm requires rmsnorm_eps to be set")
            ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        elif config.norm_type == "layernorm":
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        else:
            raise Exception(f"Unknown norm type: {config.norm_type}")

        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList([nn.Embedding(vsize, config.n_embd) for vsize in config.vocab_sizes]),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=ln_f,
            )
        )
        if speaker_emb_dim is not None:
            self.speaker_cond_pos = nn.Linear(speaker_emb_dim, config.n_embd, bias=False)

        self.lm_heads = nn.ModuleList()
        if config.target_vocab_sizes is not None:
            assert config.causal is False
        else:
            assert config.causal is True

        for vsize in config.vocab_sizes if config.target_vocab_sizes is None else config.target_vocab_sizes:
            self.lm_heads.append(nn.Linear(config.n_embd, vsize, bias=False))

        if config.target_vocab_sizes is None:
            for i in range(len(config.vocab_sizes)):
                # TODO: do we not need to take the transpose here?
                # https://paperswithcode.com/method/weight-tying
                self.lm_heads[i].weight = self.transformer.wtes[i].weight  # type: ignore
            assert len(self.lm_heads) == len(
                self.transformer.wtes  # type: ignore
            ), f"Number of heads ({len(self.lm_heads)}) must match number of one-hot embedding matrics ({len(self.transformer.wtes)})."  # type: ignore

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _mask_spk_emb_on_text(self, idx: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """
        This is in a separate function so we can test it easily.
        """
        # find index of end of text token in each sequence, then generate a binary mask
        # of shape (b, 1, t) to mask out the speaker embedding for all tokens before the end of text token.
        # Note: this does NOT mask the <end_of_text_token> token. This is important so that the first audio token predicted
        # has speaker information to use.

        # Check in channel dimension 0 as this is usually the first hierarchy where we put the text tokens.
        is_end_of_text = idx[:, 0, :] == END_OF_TEXT_TOKEN
        # use > 0, in case end_of_text_token is repeated for any reason.
        mask = (torch.cumsum(is_end_of_text, dim=-1) > 0).float()
        spk_emb = spk_emb * mask[:, :, None]

        return spk_emb

    def forward(
        self,
        idx,
        targets=None,
        speaker_embs=None,
        speaker_emb_mask=None,
        loss_reduce: Literal["mean", "none"] = "mean",
    ):
        device = idx.device
        b, num_hierarchies, t = idx.size()

        if speaker_embs is not None:
            speaker_embs = _check_speaker_emb_dims(
                speaker_embs=speaker_embs, expected_speaker_emb_dim=self.speaker_emb_dim, expected_batch_size=b
            )

        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if self.kv_cache_enabled:
            if self.kv_pos == 0:
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                self.kv_pos += t
            else:
                assert t == 1, "KV cache is only supported for single token inputs"
                pos = torch.tensor([self.kv_pos], dtype=torch.long, device=device)  # shape (1)
                self.kv_pos += 1
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        assert num_hierarchies == len(
            self.transformer.wtes
        ), f"Input tensor has {num_hierarchies} hierarchies, but model has {len(self.transformer.wtes)} set of input embeddings."

        # embed the tokens, positional encoding, and speaker embedding
        tok_emb = torch.zeros((b, t, self.config.n_embd), device=device)
        # ends up swapping (B, num_hierarchies, t) tokens -> (B, t, c) embeddings.
        for i, wte in enumerate(self.transformer.wtes):
            tok_emb += wte(idx[:, i, :])
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        spk_emb = 0.0
        if speaker_embs is not None:
            if type(speaker_embs) == list:
                assert speaker_emb_mask is None
                assert self.training is False
                assert self.spk_emb_on_text is True

                spk_emb = []
                for speaker_emb_row in speaker_embs:
                    if speaker_emb_row is not None:
                        spk_emb.append(self.speaker_cond_pos(speaker_emb_row.unsqueeze(0)))
                        assert spk_emb[-1].shape == (1, 1, self.config.n_embd), f"spk_emb[-1].shape={spk_emb[-1].shape}"
                    else:
                        spk_emb.append(torch.zeros((1, 1, self.config.n_embd), device=device, dtype=pos_emb.dtype))
                spk_emb = torch.cat(spk_emb, dim=0)

                assert (
                    spk_emb.ndim == 3 and spk_emb.shape[1] == 1 and spk_emb.shape[0] == b
                ), f"spk_emb.ndim={spk_emb.ndim}, spk_emb.shape={spk_emb.shape}, len(speaker_embs)={len(speaker_embs)}"
            else:
                speakers_embedded = self.speaker_cond_pos(speaker_embs)  # shape (b, num_examples, c)

                if speaker_emb_mask is not None:
                    spk_emb = _select_spkemb(speakers_embedded, speaker_emb_mask)
                    assert spk_emb.shape == (b, t, self.config.n_embd)
                else:
                    spk_emb = speakers_embedded
                    # if we don't have a mask, we assume that the speaker embedding is the same for all tokens
                    # then num_examples dimension just becomes the time dimension
                    assert spk_emb.ndim == 3 and spk_emb.shape[1] == 1

                if self.training and self.config.spkemb_dropout > 0.0:
                    # Remove speaker conditioning at random.
                    dropout = torch.ones_like(speakers_embedded) * (
                        torch.rand(speakers_embedded.shape[0], 1, 1, device=device) >= self.config.spkemb_dropout
                    )
                    spk_emb = torch.where(dropout == 0, torch.zeros_like(speakers_embedded), speakers_embedded)

            if self.spk_emb_on_text is False:
                assert speaker_emb_mask is None, "Not implemented for spk_emb_on_text=False"
                spk_emb = self._mask_spk_emb_on_text(idx, spk_emb)

        x = self.transformer.drop(tok_emb + pos_emb + spk_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            list_logits = [lm_head(x) for lm_head in self.lm_heads]

            losses = [
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets[:, i, :].contiguous().view(-1),
                    ignore_index=-1,
                    reduction=loss_reduce,
                )
                for i, logits in enumerate(list_logits)
            ]
            # TODO: should we do this better without stack somehow?
            losses = torch.stack(losses)
            if loss_reduce == "mean":
                losses = losses.mean()
            else:
                losses = rearrange(losses, "h (b t) -> b h t", h=len(self.lm_heads), b=b, t=t)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            if self.config.causal:
                list_logits = [
                    lm_head(x[:, [-1], :]) for lm_head in self.lm_heads
                ]  # note: using list [-1] to preserve the time dim
            else:
                list_logits = [lm_head(x) for lm_head in self.lm_heads]
            losses = None

        return list_logits, losses

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        seq_lens: Optional[list] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        speaker_embs: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        guidance_scale: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.bfloat16,
        end_of_audio_token: int = 99999,  # Dummy values will disable early termination / guidance features.
        end_of_text_token: int = 99999,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,num_hierarchies,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.dim() == 3, "idx must be a batch of sequences of hierarchical tokens"

        if self.config.causal:
            if seq_lens is None or batch_size is None:
                raise Exception("seq_lens and batch_size must be provided for causal sampling")

            return self._causal_sample(
                idx=idx,
                max_new_tokens=max_new_tokens,
                seq_lens=seq_lens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                speaker_embs=speaker_embs,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                dtype=dtype,
                end_of_audio_token=end_of_audio_token,
                end_of_text_token=end_of_text_token,
            )

        else:
            if seq_lens is not None:
                raise Exception("seq_lens is not supported yet for non-causal sampling")

            if batch_size is None:
                raise Exception("batch_size must be provided for non-causal sampling")

            if guidance_scale is not None:
                raise Exception("guidance_scale is not supported for non-causal sampling")

            if top_p is not None:
                raise Exception("top_p is not supported for non-causal sampling")

            out = []
            for start_index in tqdm.tqdm(range(0, idx.shape[0], batch_size), desc="non-causal batching"):
                end_index = min(start_index + batch_size, idx.shape[0])
                out.append(
                    self._non_causal_sample(
                        idx=idx[start_index:end_index],
                        speaker_embs=speaker_embs[start_index:end_index] if speaker_embs is not None else None,
                        temperature=temperature,
                        top_k=top_k,
                    )
                )
            return torch.cat(out, dim=0)
