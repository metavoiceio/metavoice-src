import json
import logging
import shlex
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import fastapi
import fastapi.middleware.cors
import tyro
import uvicorn
from attr import dataclass
from fastapi import Request
from fastapi.responses import Response

from fam.llm.fast_inference import TTS
from fam.llm.utils import check_audio_file

logger = logging.getLogger(__name__)


## Setup FastAPI server.
app = fastapi.FastAPI()


@dataclass
class ServingConfig:
    huggingface_repo_id: str = "metavoiceio/metavoice-1B-v0.1"
    """Absolute path to the model directory."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    port: int = 58003


# Singleton
class _GlobalState:
    config: ServingConfig
    tts: TTS


GlobalState = _GlobalState()


@dataclass(frozen=True)
class TTSRequest:
    text: str
    speaker_ref_path: Optional[str] = None
    guidance: float = 3.0
    top_p: float = 0.95
    top_k: Optional[int] = None


@app.get("/health")
async def health_check():
    return {"status": "ok"}


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
                check_audio_file(wav_path)
            else:
                # TODO: fix
                wav_path = tts_req.speaker_ref_path

            if wav_path is None:
                warnings.warn("Running without speaker reference")
                assert tts_req.guidance is None

            wav_out_path = GlobalState.tts.synthesise(
                text=tts_req.text,
                spk_ref_path=wav_path,
                top_p=tts_req.top_p,
                guidance_scale=tts_req.guidance,
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
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)

    GlobalState.config = tyro.cli(ServingConfig)
    GlobalState.tts = TTS(seed=GlobalState.config.seed)

    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*", f"http://localhost:{GlobalState.config.port}", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=GlobalState.config.port,
        log_level="info",
    )
