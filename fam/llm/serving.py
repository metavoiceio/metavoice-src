import json
import logging
import shlex
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import fastapi
import fastapi.middleware.cors
import torch
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
from fam.llm.utils import get_default_dtype, get_default_use_kv_cache

logger = logging.getLogger(__name__)


## Setup FastAPI server.
app = fastapi.FastAPI()


@dataclass
class ServingConfig:
    huggingface_repo_id: str = "metavoiceio/metavoice-1B-v0.1"
    """Absolute path to the model directory."""

    max_new_tokens: int = 864 * 2
    """Maximum number of new tokens to generate from the first stage model."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    top_k: int = 200
    """Top k for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    dtype: Literal["bfloat16", "float16", "float32", "tfloat32"] = get_default_dtype()
    """Data type to use for sampling."""

    enhancer: Optional[Literal["df"]] = "df"
    """Enhancer to use for post-processing."""

    compile: bool = False
    """Whether to compile the model using PyTorch 2.0."""

    use_kv_cache: Optional[Literal["flash_decoding", "vanilla"]] = get_default_use_kv_cache()
    """Type of kv caching to use for inference: 1) [none] no kv caching, 2) [flash_decoding] use the 
    flash decoding kernel, 3) [vanilla] use torch attention with hand implemented kv-cache."""

    port: int = 58003


# Singleton
class _GlobalState:
    spkemb_model: torch.nn.Module
    first_stage_model: Model
    second_stage_model: Model
    config: ServingConfig
    enhancer: object


GlobalState = _GlobalState()


@dataclass(frozen=True)
class TTSRequest:
    text: str
    guidance: Optional[Tuple[float, float]] = (3.0, 1.0)
    top_p: Optional[float] = 0.95
    speaker_ref_path: Optional[str] = None
    top_k: Optional[int] = None


@app.post("/tts", response_class=Response)
async def text_to_speech(req: Request):
    audiodata = await req.body()
    payload = None
    wav_out_path = None

    try:
        headers = req.headers
        payload = headers["X-Payload"]
        payload = json.loads(payload)
        tts_req = TTSRequest(**payload)
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
            if tts_req.speaker_ref_path is None:
                wav_path = _convert_audiodata_to_wav_path(audiodata, wav_tmp)
            else:
                wav_path = tts_req.speaker_ref_path
            if wav_path is None:
                warnings.warn("Running without speaker reference")
                assert tts_req.guidance is None
            wav_out_path = sample_utterance(
                tts_req.text,
                wav_path,
                GlobalState.spkemb_model,
                GlobalState.first_stage_model,
                GlobalState.second_stage_model,
                enhancer=GlobalState.enhancer,
                first_stage_ckpt_path=None,
                second_stage_ckpt_path=None,
                guidance_scale=tts_req.guidance,
                max_new_tokens=GlobalState.config.max_new_tokens,
                temperature=GlobalState.config.temperature,
                top_k=tts_req.top_k,
                top_p=tts_req.top_p,
            )
        with open(wav_out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    except Exception as e:
        # traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logger.exception(f"Error processing request {payload}")
        return Response(
            content="Something went wrong. Please try again in a few mins or contact us on Discord",
            status_code=500,
        )
    finally:
        if wav_out_path is not None:
            Path(wav_out_path).unlink(missing_ok=True)


def _convert_audiodata_to_wav_path(audiodata, wav_tmp):
    with tempfile.NamedTemporaryFile() as unknown_format_tmp:
        if unknown_format_tmp.write(audiodata) == 0:
            return None
        unknown_format_tmp.flush()

        subprocess.check_output(
            # arbitrary 2 minute cutoff
            shlex.split(f"ffmpeg -t 120 -y -i {unknown_format_tmp.name} -f wav {wav_tmp.name}")
        )

        return wav_tmp.name


if __name__ == "__main__":
    # This has to be here to avoid some weird audiocraft shenaningans messing up matplotlib
    from fam.llm.enhancers import get_enhancer

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)

    GlobalState.config = tyro.cli(ServingConfig)
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*", f"http://localhost:{GlobalState.config.port}", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    common_config = dict(
        num_samples=1,
        seed=1337,
        device=device,
        dtype=GlobalState.config.dtype,
        compile=GlobalState.config.compile,
        init_from="resume",
        output_dir=tempfile.mkdtemp(),
    )
    model_dir = snapshot_download(repo_id=GlobalState.config.huggingface_repo_id)
    config1 = InferenceConfig(
        ckpt_path=get_first_stage_path(model_dir),
        **common_config,
    )

    config2 = InferenceConfig(
        ckpt_path=get_second_stage_path(model_dir),
        **common_config,
    )

    spkemb, llm_stg1, llm_stg2 = build_models(
        config1, config2, model_dir=model_dir, device=device, use_kv_cache=GlobalState.config.use_kv_cache
    )
    GlobalState.spkemb_model = spkemb
    GlobalState.first_stage_model = llm_stg1
    GlobalState.second_stage_model = llm_stg2
    GlobalState.enhancer = get_enhancer(GlobalState.config.enhancer)

    # start server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=GlobalState.config.port,
        log_level="info",
    )
