import os
import logging
from abc import ABC
from typing import Literal, Optional

from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment

# Configure basic logging settings for the application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_wav(input_file: str, output_file: str):
    """
    Converts an audio file to WAV format.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path where the output WAV file will be saved.

    This function uses pydub.AudioSegment to read an audio file in its original format and export it as a WAV file.
    """
    try:
        logger.info("Starting convert_to_wav")
        # Extract the file format from the input file name.
        format = input_file.split(".")[-1].lower()
        # Load the audio file using its format for proper decoding.
        audio = AudioSegment.from_file(input_file, format=format)
        # Export the audio data to a new file in WAV format.
        audio.export(output_file, format="wav")
        logger.info("Finished convert_to_wav")
    except Exception as e:
        logger.error(f"Error in convert_to_wav: {e}")

def make_output_file_path(audio_file: str, tag: str, ext: Optional[str] = None) -> str:
    """
    Generates a path for the output file with an added tag and optional custom extension.

    Args:
        audio_file (str): Original path of the audio file.
        tag (str): Tag to append to the filename (before the extension).
        ext (Optional[str]): Optional custom extension for the output file. Uses original extension if None.

    Returns:
        str: Path for the output file with the specified tag and extension.
    """
    try:
        logger.info("Starting make_output_file_path")
        # Define the directory to save enhanced audio files.
        directory = "./enhanced"
        # Extract the filename from the original audio file path.
        filename = os.path.basename(audio_file)
        # Separate the filename from its extension.
        filename_without_extension = os.path.splitext(filename)[0]
        # Use the provided extension or fall back to the original extension.
        extension = ext or os.path.splitext(filename)[1]
        # Construct the output file path with the added tag and extension.
        output_file = os.path.join(directory, filename_without_extension + tag + extension)
        logger.info("Finished make_output_file_path")
        return output_file
    except Exception as e:
        logger.error(f"Error in make_output_file_path: {e}")

class BaseEnhancer(ABC):
    """
    Abstract base class for audio enhancers. Implementations should override the __call__ method.
    """
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Initializing BaseEnhancer")
            # Abstract classes cannot be instantiated.
            raise NotImplementedError
        except Exception as e:
            logger.error(f"Error in BaseEnhancer.__init__: {e}")

    def __call__(self, audio_file: str, output_file: Optional[str] = None) -> str:
        """
        Enhances an audio file. This method must be implemented by subclasses.

        Args:
            audio_file (str): Path to the input audio file.
            output_file (Optional[str]): Optional path to save the enhanced audio file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        try:
            raise NotImplementedError
        except Exception as e:
            logger.error(f"Error in BaseEnhancer.__call__: {e}")

    def get_output_file(self, audio_file: str, tag: str, ext: Optional[str] = None) -> str:
        """
        Generates a path for the output file using the specified tag and extension.

        Args:
            audio_file (str): Path to the original audio file.
            tag (str): Tag to append to the filename.
            ext (Optional[str]): Optional custom extension for the output file.

        Returns:
            str: Path for the output file with the tag and extension.
        """
        try:
            logger.info("Starting BaseEnhancer.get_output_file")
            # Generate the output file path with the specified tag and extension.
            output_file = make_output_file_path(audio_file, tag, ext=ext)
            # Ensure the directory for the output file exists.
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            logger.info("Finished BaseEnhancer.get_output_file")
            return output_file
        except Exception as e:
            logger.error(f"Error in BaseEnhancer.get_output_file: {e}")

class DFEnhancer(BaseEnhancer):
    """
    Enhancer class using the "df" enhancement algorithm. Inherits from BaseEnhancer.
    """
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Starting DFEnhancer.__init__")
            # Initialize the enhancement model and state.
            self.model, self.df_state, _ = init_df()
            logger.info("Finished DFEnhancer.__init__")
        except Exception as e:
            logger.error(f"Error in DFEnhancer.__init__: {e}")

    def __call__(self, audio_file: str, output_file: Optional[str] = None) -> str:
        """
        Enhances an audio file using the "df" enhancement algorithm.

        Args:
            audio_file (str): Path to the input audio file.
            output_file (Optional[str]): Optional path to save the enhanced audio file.

        Returns:
            str: Path to the enhanced audio file.
        """
        try:
            logger.info("Starting DFEnhancer.__call__")
            # Determine the output file path if not provided.
            output_file = output_file or self.get_output_file(audio_file, "_df")
            # Load the audio file and enhance it using the "df" algorithm.
            audio, _ = load_audio(audio_file, sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
            # Save the enhanced audio to the specified output file.
            save_audio(output_file, enhanced, self.df_state.sr())
            logger.info("Finished DFEnhancer.__call__")
            return output_file
        except Exception as e:
            logger.error(f"Error in DFEnhancer.__call__: {e}")

def get_enhancer(enhancer_name: Literal["df"]) -> BaseEnhancer:
    """
    Factory function to get an enhancer instance based on the enhancer name.

    Args:
        enhancer_name (Literal["df"]): Name of the enhancer to instantiate.

    Returns:
        BaseEnhancer: Instance of the specified enhancer.

    Raises:
        ValueError: If an unknown enhancer name is provided.
    """
    try:
        logger.info("Starting get_enhancer")
        # Instantiate the appropriate enhancer based on the provided name.
        if enhancer_name == "df":
            enhancer = DFEnhancer()
        else:
            # Raise an error for unsupported enhancer names.
            raise ValueError(f"Unknown enhancer name: {enhancer_name}")
        logger.info("Finished get_enhancer")
        return enhancer
    except Exception as e:
        logger.error(f"Error in get_enhancer: {e}")
