from typing import Optional

import torch
from torch.nn import functional as F


class NonCausalInferenceMixin:
    """
    Mixin class for non-causal inference in a language model.

    This class provides methods for performing non-causal sampling using a language model.
    """

    @torch.no_grad()
    def _non_causal_sample(
        self, *, idx: torch.Tensor, speaker_embs: Optional[torch.Tensor], temperature: float, top_k: int
    ):
        """
        Perform non-causal sampling.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, num_in_hierarchies, sequence_length).
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings tensor of shape (batch_size, embedding_size).
            temperature (float): Temperature parameter for scaling the logits.
            top_k (int): Number of top options to consider.

        Returns:
            torch.Tensor: Sampled output tensor of shape (batch_size, num_out_hierarchies, sequence_length).
        """
        b, c, t = idx.size()
        assert t == self.config.block_size, f"input size {t} != config.block_size {self.config.block_size}"
        # forward the model to get the logits for the index in the sequence
        list_logits, _ = self(idx, speaker_embs=speaker_embs)  # c x (b, t, vocab_size)

        # scale by desired temperature
        list_logits = [logits / temperature for logits in list_logits]  # c x (b, t, vocab_size)

        # optionally crop the logits to only the top k options
        if top_k is not None:
            for i in range(len(list_logits)):
                logits = list_logits[i]  # (b, t, vocab_size)

                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (b, t, top_k)
                logits[logits < v[:, :, [-1]]] = -float("Inf")
                list_logits[i] = logits  # (b, t, vocab_size)
                assert logits.shape[0] == b and logits.shape[1] == t

        # apply softmax to convert logits to (normalized) probabilities
        # TODO: check shapes here!
        probs = [F.softmax(logits, dim=-1) for logits in list_logits]  # c x (b, t, top_k)
        assert probs[0].shape[0] == b and probs[0].shape[1] == t

        # TODO: output shape is as expected
        outs = []
        for b_prob in probs:  # c x (b, t, top_k) -> (b, t, top_k)
            out = [
                torch.multinomial(prob, num_samples=1).transpose(0, 1).unsqueeze(0) for prob in b_prob
            ]  # b x (t, top_k) -> b x (t, 1) -> b x (1, t) -> b x (1, 1, t)
            assert len(out) == b and out[0].shape[0] == 1 and out[0].shape[1] == 1 and out[0].shape[2] == t
            out = torch.cat(out, dim=0)  # (b, 1, t)
            assert out.shape[0] == b and out.shape[1] == 1 and out.shape[2] == t
            outs.append(out)

        out = torch.cat(outs, dim=1)  # (b, c, t)
        assert out.shape[0] == b and out.shape[2] == t

        return out
