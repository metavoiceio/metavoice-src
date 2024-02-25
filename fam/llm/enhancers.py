import os
from abc import ABC
from typing import Literal, Optional

from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment


def convert_to_wav(input_file: str, output_file: str):
    """Convert an audio file to WAV format

    Args:
        input_file (str): path to input audio file
        output_file (str): path to output WAV file

    """
    # Detect the format of the input file
    format = input_file.split(".")[-1].lower()

    # Read the audio file
    audio = AudioSegment.from_file(input_file, format=format)

    # Export as WAV
    audio.export(output_file, format="wav")


def make_output_file_path(audio_file: str, tag: str, ext: Optional[str] = None) -> str:
    """Generate the output file path

    Args:
        audio_file (str): path to input audio file
        tag (str): tag to append to the output file name
        ext (str, optional): extension of the output file. Defaults to None.

    Returns:
        str: path to output file
    """

    directory = "./enhanced"
    # Get the name of the input file
    filename = os.path.basename(audio_file)

    # Get the name of the input file without the extension
    filename_without_extension = os.path.splitext(filename)[0]

    # Get the extension of the input file
    extension = ext or os.path.splitext(filename)[1]

    # Generate the output file path
    output_file = os.path.join(directory, filename_without_extension + tag + extension)

    return output_file


class BaseEnhancer(ABC):
    """Base class for audio enhancers"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, audio_file: str, output_file: Optional[str] = None) -> str:
        raise NotImplementedError

    def get_output_file(self, audio_file: str, tag: str, ext: Optional[str] = None) -> str:
        output_file = make_output_file_path(audio_file, tag, ext=ext)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return output_file


class DFEnhancer(BaseEnhancer):
    def __init__(self, *args, **kwargs):
        self.model, self.df_state, _ = init_df()

    def __call__(self, audio_file: str, output_file: Optional[str] = None) -> str:
        output_file = output_file or self.get_output_file(audio_file, "_df")

        audio, _ = load_audio(audio_file, sr=self.df_state.sr())

        enhanced = enhance(self.model, self.df_state, audio)

        save_audio(output_file, enhanced, self.df_state.sr())

        return output_file


def get_enhancer(enhancer_name: Literal["df"]) -> BaseEnhancer:
    """Get an audio enhancer

    Args:
        enhancer_name (Literal["df"]): name of the audio enhancer

    Raises:
        ValueError: if the enhancer name is not recognised

    Returns:
        BaseEnhancer: audio enhancer
    """

    if enhancer_name == "df":
        import warnings

        warnings.filterwarnings(
            "ignore",
            message='"sinc_interpolation" resampling method name is being deprecated and replaced by "sinc_interp_hann" in the next release. The default behavior remains unchanged.',
        )
        return DFEnhancer()
    else:
        raise ValueError(f"Unknown enhancer name: {enhancer_name}")
