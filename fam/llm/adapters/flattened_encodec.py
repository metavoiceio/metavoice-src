from fam.llm.adapters.base import BaseDataAdapter


class FlattenedInterleavedEncodec2Codebook(BaseDataAdapter):
    def __init__(self, end_of_audio_token):
        self._end_of_audio_token = end_of_audio_token

    def decode(self, tokens: list[list[int]]) -> tuple[list[int], list[list[int]]]:
        assert len(tokens) == 1
        tokens = tokens[0]

        text_ids = []
        extracted_audio_ids = [[], []]

        for t in tokens:
            if t < self._end_of_audio_token:
                extracted_audio_ids[0].append(t)
            elif t >= self._end_of_audio_token and t < 2 * self._end_of_audio_token:
                extracted_audio_ids[1].append(t - self._end_of_audio_token)
            # We ignore t = 2 * self._end_of_audio_token, as it is the end of audio token
            elif t > 2 * self._end_of_audio_token:
                text_ids.append(t)

        if len(set([len(x) for x in extracted_audio_ids])) != 1:
            min_len = min([len(x) for x in extracted_audio_ids])
            max_len = max([len(x) for x in extracted_audio_ids])
            print("WARNING: Number of tokens at each hierarchy must be of the same length!")
            print(f"Truncating to min length of {min_len} tokens from {max_len} max.")
            print([len(x) for x in extracted_audio_ids])
            extracted_audio_ids = [x[:min_len] for x in extracted_audio_ids]

        return text_ids[:-1], extracted_audio_ids

    def encode(self, text_tokens: list[int], audio_tokens: list[list[int]]):
        """
        Performs the required combination and padding as needed.
        """
        raise NotImplementedError
