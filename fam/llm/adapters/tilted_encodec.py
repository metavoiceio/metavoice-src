from fam.llm.adapters.base import BaseDataAdapter


class TiltedEncodec(BaseDataAdapter):
    def __init__(self, end_of_audio_token):
        self._end_of_audio_token = end_of_audio_token

    def decode(self, tokens: list[list[int]]) -> tuple[list[int], list[list[int]]]:
        assert len(tokens) > 1

        text_ids = []
        extracted_audio_ids = []

        extracted_audio_ids.append([])
        # Handle first hierarchy as special case as it contains text tokens as well
        # TODO: maybe it doesn't need special case, and can be handled on it's own :)
        for t in tokens[0]:
            if t > self._end_of_audio_token:
                text_ids.append(t)
            elif t < self._end_of_audio_token:
                extracted_audio_ids[0].append(t)

        # Handle the rest of the hierarchies
        for i in range(1, len(tokens)):
            token_hierarchy_ids = tokens[i]
            extracted_audio_ids.append([])
            for t in token_hierarchy_ids:
                if t < self._end_of_audio_token:
                    extracted_audio_ids[i].append(t)

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
