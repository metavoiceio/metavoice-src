import json
import logging
import shlex
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import fastapi
import fastapi.middleware.cors
import torch
import torchaudio
from torchaudio.transforms import Resample

import tyro
import uvicorn
from attr import dataclass
from fastapi import Request
from fastapi.responses import Response
from huggingface_hub import snapshot_download

from fam.llm.sample import (
    InferenceConfig,
    Model,
    build_models,
    get_first_stage_path,
    get_second_stage_path,
    sample_utterance,
)

# Configure the logger for this module.
logger = logging.getLogger(__name__)

# Initialize a FastAPI app instance.
app = fastapi.FastAPI()

@dataclass
class ServingConfig:
    """
    Configuration class for server settings and model parameters.

    Attributes:
        huggingface_repo_id (str): Identifier for the Hugging Face repository containing the model.
        max_new_tokens (int): Maximum number of new tokens to generate (default is a preset value).
        temperature (float): Sampling temperature for generating predictions (default is 1.0).
        top_k (int): Top K sampling parameter (default is 200).
        seed (int): Random seed for reproducibility (default is 1337).
        dtype (Literal): Data type for model computations (default is 'bfloat16').
        enhancer (Optional[Literal]): Optional audio enhancer name, e.g., 'df' (default is 'df').
        compile (bool): Whether to compile the model for optimization (default is False).
        port (int): Port number for the FastAPI server (default is 58003).
    """
    huggingface_repo_id: str
    max_new_tokens: int = 864 * 2
    temperature: float = 1.0
    top_k: int = 200
    seed: int = 1337
    dtype: Literal["bfloat16", "float16", "float32", "tfloat32"] = "bfloat16"
    enhancer: Optional[Literal["df"]] = "df"
    compile: bool = False
    port: int = 58003

class _GlobalState:
    """
    Global state for the server, holding models, configuration, and enhancer instance.

    Attributes:
        spkemb_model (torch.nn.Module): Speaker embedding model.
        first_stage_model (Model): First stage model of the TTS pipeline.
        second_stage_model (Model): Second stage model of the TTS pipeline.
        config (ServingConfig): Configuration for the server and model parameters.
        enhancer (object): Audio enhancer instance, if any.
    """
    spkemb_model: torch.nn.Module
    first_stage_model: Model
    second_stage_model: Model
    config: ServingConfig
    enhancer: object

# Initialize the global state.
GlobalState = _GlobalState()

@dataclass(frozen=True)
class TTSRequest:
    """
    Data class for incoming text-to-speech (TTS) requests.

    Attributes:
        text (str): Text to synthesize into speech.
        guidance (Optional[Union[float, Tuple[float, float]]]): Guidance scales for the TTS models.
        top_p (Optional[float]): Nucleus sampling parameter.
        speaker_ref_path (Optional[str]): Path to a reference audio file for speaker embedding.
        top_k (Optional[int]): Top K sampling parameter for the TTS model.
    """
    text: str
    guidance: Optional[Union[float, Tuple[float, float]]] = (3.0, 1.0)
    top_p: Optional[float] = 0.95
    speaker_ref_path: Optional[str] = None
    top_k: Optional[int] = None

@app.post("/tts", response_class=Response)
async def text_to_speech(req: Request):
    """
    Endpoint for processing text-to-speech requests.

    Args:
        req (Request): FastAPI request object containing the TTS request data.

    Returns:
        Response: FastAPI response object with the synthesized speech as audio data.
    """
    audiodata = await req.body()
    payload = None
    wav_out_path = None

    try:
        headers = req.headers
        payload = headers["X-Payload"]
        payload = json.loads(payload)
        tts_req = TTSRequest(**payload)

        # Unpack guidance values if provided as a tuple or use single value for both stages.
        if isinstance(tts_req.guidance, tuple):
            first_guidance, second_guidance = tts_req.guidance
        elif isinstance(tts_req.guidance, (float, int)):
            first_guidance = second_guidance = tts_req.guidance
        else:
            first_guidance = second_guidance = None

        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
            # Convert incoming audio data to a WAV file if no speaker reference is provided.
            if tts_req.speaker_ref_path is None:
                wav_path = _convert_audiodata_to_wav_path(audiodata, wav_tmp)
            else:
                wav_path = tts_req.speaker_ref_path
            # If no WAV path could be determined, proceed without speaker reference.
            if wav_path is None:
                warnings.warn("Running without speaker reference")
                assert tts_req.guidance is None
            # Use the TTS pipeline to synthesize the requested utterance.
            wav_out_path = sample_utterance(
                tts_req.text,
                wav_path,
                GlobalState.spkemb_model,
                GlobalState.first_stage_model,
                GlobalState.second_stage_model,
                enhancer=GlobalState.enhancer,
                first_stage_ckpt_path=None,
                second_stage_ckpt_path=None,
                guidance_scale=(first_guidance, second_guidance),
                max_new_tokens=GlobalState.config.max_new_tokens,
                temperature=GlobalState.config.temperature,
                top_k=tts_req.top_k,
                top_p=tts_req.top_p,
            )
        # Return the synthesized speech as a WAV file.
        with open(wav_out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    except Exception as e:
        logger.exception(f"Error processing request {payload}")
        # Return an error response in case of failure.
        return Response(
            content="Something went wrong. Please try again in a few mins or contact us on Discord",
            status_code=500,
        )
    finally:
        # Clean up the generated WAV file after serving the response.
        if wav_out_path is not None:
            Path(wav_out_path).unlink(missing_ok=True)

def _convert_audiodata_to_wav_path(audiodata, wav_tmp_path):
    """
    Converts incoming audio data to a WAV file and returns the path to the file.

    Args:
        audiodata: Raw audio data from the request.
        wav_tmp_path: Temporary file path for intermediate storage.

    Returns:
        The path to the converted WAV file.
    """
    with tempfile.NamedTemporaryFile(delete=False) as unknown_format_tmp:
        # Write the incoming audio data to a temporary file.
        if unknown_format_tmp.write(audiodata) == 0:
            return None
        unknown_format_tmp.flush()
        unknown_format_tmp.close()  # Explicitly close the file to ensure data is written.

        # Generate a temporary path for the output WAV file.
        output_path = tempfile.mktemp(suffix=".wav")

        # Use ffmpeg to convert the audio data to WAV format.
        subprocess.check_output(
            shlex.split(f'ffmpeg -t 120 -y -i "{unknown_format_tmp.name}" -f wav "{output_path}"')
        )

        return output_path

if __name__ == "__main__":
    from fam.llm.enhancers import get_enhancer

    # Set logging levels for all loggers to INFO.
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)

    # Parse command-line arguments to configure the server.
    GlobalState.config = tyro.cli(ServingConfig)

    # Setup CORS middleware for the FastAPI app.
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*", f"http://localhost:{GlobalState.config.port}", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Determine the device to use based on CUDA availability.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Common configuration settings for the models.
    common_config = dict(
        num_samples=1,
        seed=1337,
        device=device,
        dtype=GlobalState.config.dtype,
        compile=GlobalState.config.compile,
        init_from="resume",
        output_dir=tempfile.mkdtemp(),
    )

    # Download the model from Hugging Face Hub.
    model_dir = snapshot_download(repo_id=GlobalState.config.huggingface_repo_id)

    # Configure and build the first and second stage models.
    config1 = InferenceConfig(
        ckpt_path=get_first_stage_path(model_dir),
        **common_config,
    )

    config2 = InferenceConfig(
        ckpt_path=get_second_stage_path(model_dir),
        **common_config,
    )

    # Initialize the models and enhancer.
    spkemb, llm_stg1, llm_stg2 = build_models(
        config1, config2, model_dir=model_dir, device=device, use_kv_cache="flash_decoding"
    )
    GlobalState.spkemb_model = spkemb
    GlobalState.first_stage_model = llm_stg1
    GlobalState.second_stage_model = llm_stg2
    GlobalState.enhancer = get_enhancer(GlobalState.config.enhancer)

    # Start the FastAPI server.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=GlobalState.config.port,
        log_level="info",
    )
