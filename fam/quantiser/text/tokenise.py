import tiktoken


class TrainedBPETokeniser:
    def __init__(self, name, pat_str, mergeable_ranks, special_tokens, offset=None) -> None:
        self.tokenizer = tiktoken.Encoding(
            name=name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        self.offset = offset

    def encode(self, text: str) -> list[int]:
        # note: we add a end of text token!
        tokens = self.tokenizer.encode(text) + [self.tokenizer.eot_token]
        if self.offset is not None:
            tokens = [x + self.offset for x in tokens]

        return tokens

    def decode(self, tokens: list[int]):
        if self.offset is not None:
            tokens = [x - self.offset for x in tokens]
        return self.tokenizer.decode(tokens)

    @property
    def eot_token(self):
        if self.offset is not None:
            return self.tokenizer.eot_token + self.offset
        else:
            return self.tokenizer.eot_token
