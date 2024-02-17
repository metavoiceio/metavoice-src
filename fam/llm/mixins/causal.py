from typing import Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.nn import functional as F


def top_p_sample(prob_dist: torch.Tensor, top_p: float):
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True, dim=-1)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)  # (b, vocab_size)

    sorted_indices_to_remove = cum_sum_probs > top_p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_indices_to_remove = sorted_indices_to_remove.bool()

    # replace probs to be removed with 0 in the sorted_probs
    sorted_probs[sorted_indices_to_remove] = 0

    # reverse the sorting process
    reversed_indices = torch.argsort(sorted_indices)
    prob_dist = torch.gather(sorted_probs, -1, reversed_indices)

    # normalize
    prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)

    return prob_dist


class CausalInferenceMixin:
    """
    Mixin class for performing inference in a causal language model.

    This mixin provides methods for predicting the next token in a sequence, sampling from the model,
    and applying token prediction masks.

    Attributes:
        None

    Methods:
        _sample_next_token: Predicts the next token in the sequence.
        _create_token_pred_mask: Creates a token prediction mask based on sequence lengths.
        _apply_token_pred_mask: Applies a token prediction mask to the next token predictions.
        _sample_batch: Samples a batch of tokens from the model.
        _sort_for_batching: Sorts the input sequences for efficient batching.
        _causal_sample: Generates a sequence of tokens using causal sampling.

    """

    @torch.no_grad()
    def _sample_next_token(
        self,
        *,
        idx: torch.Tensor,
        speaker_embs: Optional[torch.Tensor],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        guidance_scale: Optional[Tuple[float, float]],
    ) -> torch.Tensor:
        """
        Predict the next token in the sequence.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Next index in the sequence after sampling. Shape: (batch, num_hierarchies).
        """
        if top_k is not None and top_p is not None:
            raise ValueError("Only one of top_k and top_p can be set")

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:, :, -self.config.block_size :]

        # forward the model to get the logits for the index in the sequence
        list_logits, _ = self(
            idx_cond, speaker_embs=speaker_embs
        )  # list with len num_hierarchies of (b,1,vocab_size) tensors

        if guidance_scale is not None:
            spkemb_guidance_scale, prompt_guidance_scale = guidance_scale
            assert spkemb_guidance_scale >= 1
            assert prompt_guidance_scale >= 1
            base_scale = spkemb_guidance_scale + prompt_guidance_scale - 1

            for i, logits in enumerate(list_logits):
                if prompt_guidance_scale > 1:
                    logits_cond, logits_uncond_spkemb, logits_uncond_prompt = logits.split(logits.shape[0] // 3, dim=0)
                else:
                    logits_cond, logits_uncond_spkemb = logits.split(logits.shape[0] // 2, dim=0)
                    logits_uncond_prompt = 0
                list_logits[i] = (
                    (base_scale) * logits_cond
                    + (1 - spkemb_guidance_scale) * logits_uncond_spkemb
                    + (1 - prompt_guidance_scale) * logits_uncond_prompt
                )

        # pluck the logits at the final step and scale by desired temperature
        list_logits = [
            logits[:, -1, :] / temperature for logits in list_logits
        ]  # list with len num_hierarchies of (b,vocab_size) tensors

        # optionally crop the logits to only the top k options
        if top_k is not None:
            for i in range(len(list_logits)):
                logits = list_logits[i]
                v, _ = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )  # returns a descending sorted list of values and indices of top_k values
                logits[logits < v[:, [-1]]] = -float("Inf")  # set all logits below the smallest top_k value to -Inf
                list_logits[i] = logits

        # apply softmax to convert logits to (normalized) probabilities
        probs = [
            F.softmax(logits, dim=-1) for logits in list_logits
        ]  # list of len num_hierarchies of (b,vocab_size) tensors

        if top_p is not None:
            for i in range(len(probs)):
                probs[i] = top_p_sample(probs[i], top_p)

        # sample from the distribution
        idx_next = [
            torch.multinomial(prob, num_samples=1) for prob in probs
        ]  # list of len num_hierarchies of (b,1) tensors
        idx_next = torch.cat(idx_next, dim=-1)  # (b, num_hierarchies) tensor

        return idx_next  # (b, num_hierarchies) tensor

    @torch.no_grad()
    def _create_token_pred_mask(self, idx: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        """
        Creates a token prediction mask based on sequence lengths.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.

        Returns:
            torch.Tensor: Token prediction mask of shape (batch, time).
        """
        token_pred_mask = torch.zeros((idx.shape[0], idx.shape[-1]), dtype=torch.bool, device=idx.device)
        for i in range(len(seq_lens)):
            token_pred_mask[i, : seq_lens[i]] = True

        assert (token_pred_mask[:, : min(seq_lens)] == 1).all()

        return token_pred_mask

    @torch.no_grad()
    def _apply_token_pred_mask(
        self, *, idx_next: torch.Tensor, orig_input_at_t: torch.Tensor, token_pred_mask_at_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a token prediction mask to the next token predictions.

        Args:
            idx_next (torch.Tensor): Next token predictions of shape (batch, num_hierarchies).
            orig_input_at_t (torch.Tensor): Original input at time step t of shape (batch, num_hierarchies).
            token_pred_mask_at_t (torch.Tensor): Token prediction mask at time step t of shape (batch, 1).

        Returns:
            torch.Tensor: Updated next token predictions after applying the token prediction mask.
        """
        idx_next = idx_next * (~token_pred_mask_at_t) + orig_input_at_t * token_pred_mask_at_t

        return idx_next

    @torch.no_grad()
    def _sample_batch(
        self,
        *,
        idx: torch.Tensor,
        max_new_tokens: int,
        seq_lens: list[int],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        speaker_embs: Optional[torch.Tensor],
        guidance_scale: Optional[Tuple[float, float]],
        end_of_audio_token: int,
        end_of_text_token: int,
    ):
        """
        Samples a batch of tokens from the model.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Generated sequence indices of shape (batch, num_hierarchies, time).
        """
        assert max(seq_lens) <= idx.shape[-1]
        token_pred_mask = self._create_token_pred_mask(idx, seq_lens)
        input = torch.clone(idx)

        min_seq_lens = min(seq_lens)
        idx = idx[:, :, :min_seq_lens]
        idx_out = torch.full(
            (idx.shape[0], idx.shape[1], idx.shape[2] + max_new_tokens),
            end_of_audio_token,
            dtype=idx.dtype,
            device=idx.device,
        )
        idx_out[:, :, :min_seq_lens] = idx
        terminated = idx.new_zeros(idx.shape[0], dtype=torch.bool)

        if guidance_scale is not None:
            _, prompt_guidance_scale = guidance_scale
            if speaker_embs is None:
                raise Exception("Guidance is only supported for conditional models")

            # create speaker embeddings equivalent to the batch size, filling with None
            # for second half to do unconditional generation.
            speaker_embs = (
                list(speaker_embs)
                + [None] * (speaker_embs.shape[0])
                + (list(speaker_embs) if prompt_guidance_scale > 1 else [])
            )

        for timestep in tqdm.tqdm(range(min_seq_lens, min_seq_lens + max_new_tokens), desc="tokens: "):
            if terminated.all():
                break
            if (self.kv_cache_enabled is True) and (timestep > min_seq_lens):
                idx_input = idx_out[:, :, [timestep - 1]]
            else:
                idx_input = idx_out[:, :, :timestep]

            if guidance_scale is not None:
                _, prompt_guidance_scale = guidance_scale
                # TODO: fix: will cause a problem with kv-caching as it's not expecting larger batch-size.
                if timestep == min_seq_lens:
                    print("[hack!!!!] Guidance is on, so we're doubling/tripling batch size!")

                # replicate idx in the batch dimension
                idx_input = (
                    idx_input.unsqueeze(0)
                    .repeat(3 if prompt_guidance_scale > 1 else 2, 1, 1, 1)
                    .reshape(-1, idx_input.shape[1], idx_input.shape[2])
                )

                if prompt_guidance_scale > 1:
                    idx_input_uncond = idx_input[idx_input.shape[0] // 3 * 2 :]
                    idx_input_uncond = idx_input_uncond.view(-1)
                    # Replace all text tokens with endoftext token
                    idx_input_uncond[idx_input_uncond > end_of_audio_token] = end_of_text_token

            idx_next = self._sample_next_token(
                idx=idx_input,
                speaker_embs=speaker_embs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale,
            )  # (b, num_hierarchies)

            assert idx_next.shape[0] == idx.shape[0]

            if timestep < token_pred_mask.shape[-1]:
                idx_next = self._apply_token_pred_mask(
                    idx_next=idx_next,
                    orig_input_at_t=input[:, :, timestep],
                    token_pred_mask_at_t=token_pred_mask[:, [timestep]],
                )
            is_endofaudio = (idx_next == end_of_audio_token).any(dim=-1)  # shape: b
            terminated = terminated | is_endofaudio
            idx_next[terminated] = end_of_audio_token
            # append sampled index to the running sequence and continue
            idx_out[:, :, timestep] = idx_next

        return idx_out

    @torch.no_grad()
    def _sort_for_batching(
        self,
        *,
        idx: torch.Tensor,
        seq_lens: list[int],
        speaker_embs: Optional[torch.Tensor],
        batch_size: int,
        max_new_tokens: int,
    ) -> Tuple[list[int], list[int], torch.Tensor, list[int], Optional[torch.Tensor], int]:
        """
        Sorts the input sequences for efficient batching.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            batch_size (int): Batch size for sampling. idx is split into batches of this size for sampling.
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).

        Returns:
            Tuple[list[int], list[int], torch.Tensor, list[int], Optional[torch.Tensor], int]:
                - sorted_indices (list[int]): List of indices of the input sequences that transform it into sorted order.
                - invert_sorted_indices (list[int]): List of indices to invert the sorted sequences back to the original order.
                - idx (torch.Tensor): Input sequence indices in sorted order.
                - seq_lens (list[int]): Sequence lengths in sorted order.
                - speaker_embs (Optional[torch.Tensor]): speaker embeddings in sorted order.
                - max_token_len (int): Effective maximum number of tokens to generate.
        """
        assert len(seq_lens) == idx.shape[0]
        assert max(seq_lens) <= idx.shape[-1]

        sorted_indices = np.argsort(seq_lens)
        inverted_sorted_indices = np.zeros(len(seq_lens), dtype=np.int32)
        inverted_sorted_indices[sorted_indices] = np.arange(len(seq_lens), dtype=np.int32)

        idx = idx[sorted_indices]
        seq_lens = [seq_lens[i] for i in sorted_indices]
        speaker_embs = speaker_embs[sorted_indices] if speaker_embs is not None else None
        max_token_len = 0

        # figure out effective max_tokens to generate
        for start_index in range(0, len(seq_lens), batch_size):
            end_index = min(start_index + batch_size, len(seq_lens))
            batch_seq_lens = seq_lens[start_index:end_index]
            # random heuristic...
            # # TODO: fix!
            max_token_len = max(max_token_len, min(batch_seq_lens) + max_new_tokens)

        return sorted_indices, inverted_sorted_indices, idx, seq_lens, speaker_embs, max_token_len

    @torch.no_grad()
    def _causal_sample(
        self,
        *,
        idx: torch.Tensor,
        max_new_tokens: int,
        seq_lens: list[int],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        speaker_embs: Optional[torch.Tensor],
        batch_size: int,
        guidance_scale: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.bfloat16,
        end_of_audio_token: int,
        end_of_text_token: int,
    ) -> torch.Tensor:
        """
        Generates a sequence of tokens using causal sampling.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            batch_size (int): Batch size for sampling. idx is split into batches of this size for sampling.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Generated sequence indices of shape (batch, num_hierarchies, time).
        """
        (
            _,
            invert_sorted_indices,
            idx,
            seq_lens,
            speaker_embs,
            max_token_len,
        ) = self._sort_for_batching(
            idx=idx, seq_lens=seq_lens, speaker_embs=speaker_embs, batch_size=batch_size, max_new_tokens=max_new_tokens
        )

        return_idx = torch.zeros((len(seq_lens), idx.size(1), max_token_len), dtype=torch.long, device=idx.device)

        for start_index in tqdm.tqdm(range(0, len(seq_lens), batch_size), desc="batch: "):
            end_index = min(start_index + batch_size, len(seq_lens))

            kv_batch_size = end_index - start_index
            if guidance_scale is not None:
                if guidance_scale[1] > 1:
                    kv_batch_size = 3 * kv_batch_size
                else:
                    kv_batch_size = 2 * kv_batch_size

            if self.kv_cache_enabled:
                self.empty_kv_cache(
                    batch_size=kv_batch_size,
                    kv_cache_maxlen=self.config.block_size,
                    dtype=dtype,
                )

            batch_seq_lens = seq_lens[start_index:end_index]
            batch_max_new_tokens = max_token_len - min(batch_seq_lens)

            batch_idx = idx[start_index:end_index]
            batch_speaker_embs = speaker_embs[start_index:end_index] if speaker_embs is not None else None

            batch_idx = self._sample_batch(
                idx=batch_idx,
                max_new_tokens=batch_max_new_tokens,
                seq_lens=batch_seq_lens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                speaker_embs=batch_speaker_embs,
                guidance_scale=guidance_scale,
                end_of_audio_token=end_of_audio_token,
                end_of_text_token=end_of_text_token,
            )
            return_idx[start_index:end_index] = batch_idx

        return return_idx[invert_sorted_indices]

    def empty_kv_cache(self, *, batch_size: int, kv_cache_maxlen: int, dtype: torch.dtype):
        """
        Empties key-value (KV) cache for causal attention.

        Args:
            batch_size (int): The batch size.
            kv_cache_maxlen (int): The maximum length of the KV cache.
            dtype (torch.dtype): The data type of the KV cache.

        Raises:
            Exception: If KV cache is enabled for non-causal attention.

        """
        if self.kv_cache_enabled is False:
            raise Exception("KV cache is not enabled")
        if self.config.causal is False:
            raise Exception("KV cache is not supported for non-causal attention")

        self.kv_pos = 0
        for block in self.transformer.h:
            block.attn.empty_kv_cache(batch_size=batch_size, kv_cache_maxlen=kv_cache_maxlen, dtype=dtype)

    def enable_kv_cache(self):
        """
        Enables key-value (KV) cache for causal attention.

        Raises:
            Exception: If KV cache is enabled for non-causal attention.

        """
        if self.config.causal is False:
            raise Exception("KV cache is not supported for non-causal attention")

        self.kv_cache_enabled = True
        for block in self.transformer.h:
            block.attn.kv_cache_enabled = True

    def disable_kv_cache(self):
        """
        Disables the key-value cache for the transformer and all its blocks.
        """
        self.kv_cache_enabled = False
        for block in self.transformer.h:
            block.attn.kv_cache_enabled = False
            block.attn.kv_cache = None
            block.attn.kv_cache_first_empty_index = 0

    @torch.no_grad()
    def _slow_causal_sampling_loop(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        speaker_embs: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
    ):
        """
        Old non-batched version of causal sampling. Kept for testing / reference.

        Take a conditioning sequence of indices idx (LongTensor of shape (b,n_head,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.dim() == 3, "idx must be a batch of sequences of hierarchical tokens"
        assert idx.size(0) == 1, "can only do one sequence at a time for now"
        assert top_p is None, "nucleus sampling not supported yet with _slow_causal_sampling_loop"

        if self.config.causal is not True:
            raise Exception("Causal sampling is only supported for causal models")

        if self.kv_cache_enabled:
            print("!!!! USING KV-CACHING ASSUMED TORCH.BFLOAT16")
            self.empty_kv_cache(
                batch_size=1,
                kv_cache_maxlen=self.config.block_size,
                dtype=torch.bfloat16,
            )

        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:, -self.config.block_size :]

            if self.kv_cache_enabled:
                if i > 0:
                    idx_cond = idx_cond[:, :, -1:]

            # forward the model to get the logits for the index in the sequence
            list_logits, _ = self(idx_cond, speaker_embs=speaker_embs)

            if guidance_scale is not None:
                # we've already checked that kv-caching is not switched on
                # so this should be ok.
                list_logits_uncond, _ = self(idx_cond, speaker_embs=None)
                list_logits = [
                    (guidance_scale) * logits + (1 - guidance_scale) * logits_uncond
                    for logits, logits_uncond in zip(list_logits, list_logits_uncond)
                ]

            # pluck the logits at the final step and scale by desired temperature
            list_logits = [logits[:, -1, :] / temperature for logits in list_logits]

            # optionally crop the logits to only the top k options
            if top_k is not None:
                for i in range(len(list_logits)):
                    logits = list_logits[i]
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                    list_logits[i] = logits

            # apply softmax to convert logits to (normalized) probabilities
            probs = [F.softmax(logits, dim=-1) for logits in list_logits]
            # sample from the distribution
            idx_next = torch.tensor(
                [torch.multinomial(prob, num_samples=1) for prob in probs], device=idx.device
            )  # (c, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next.unsqueeze(0).unsqueeze(-1)), dim=2)

        return idx
