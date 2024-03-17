from functools import partial
from typing import Any, Callable, Literal, Optional

import numpy as np

AudioTokenModeT = Literal["flattened_interleaved"]
CombinerWithOffsetFuncT = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
CombinerFuncT = Callable[[np.ndarray, np.ndarray], np.ndarray]


def combine_tokens_flattened_interleaved(
    audio_tokens: np.ndarray, text_tokens: np.ndarray, second_hierarchy_flattening_offset: int
) -> np.ndarray:
    """
    Flattens & interleaves first 2 of the audio token hierarchies. Note that the tokens for the second hierarchy
    are also offset by second_hierarchy_flattening_offset as part of this transform to avoid conflict with values for the
    first hierarchy.
    """
    assert np.issubdtype(audio_tokens.dtype, np.integer)
    assert np.issubdtype(text_tokens.dtype, np.integer)

    num_hierarchies = audio_tokens.shape[0]
    assert num_hierarchies >= 2, f"Unexpected number of hierarchies: {num_hierarchies}. Expected at least 2."

    # choosing -5 so that we can't get error!
    interleaved_audio_tokens = np.full((len(audio_tokens[0]) + len(audio_tokens[1]),), -5)
    interleaved_audio_tokens[::2] = audio_tokens[0]
    interleaved_audio_tokens[1::2] = audio_tokens[1] + second_hierarchy_flattening_offset

    tokens = np.concatenate([text_tokens, interleaved_audio_tokens])

    return np.expand_dims(tokens, axis=0)


def get_params_for_mode(
    audio_token_mode: AudioTokenModeT, num_max_audio_tokens_timesteps: Optional[int] = None
) -> dict[str, Any]:
    if audio_token_mode == "flattened_interleaved":
        return {
            "text_tokenisation_offset": 1024 * 2 + 1,
            "pad_token": 1024 * 2,
            "ctx_window": num_max_audio_tokens_timesteps * 2 if num_max_audio_tokens_timesteps else None,
            "second_hierarchy_flattening_offset": 1024,
            # TODO: fix the repeat of `second_hierarchy_flattening_offset`
            "combine_func": partial(
                combine_tokens_flattened_interleaved,
                second_hierarchy_flattening_offset=1024,
            ),
        }
    else:
        raise Exception(f"Unknown mode {audio_token_mode}")
