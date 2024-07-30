import signal
import queue
from fastapi import FastAPI, WebSocket
import random
from starlette.websockets import WebSocketDisconnect
import traceback
import torch
import time
import os
from fam.llm.fast_inference import TTS, flush_queue

app = FastAPI()
model = None
texts = [
    "This is an example of speech synthesis using MetaVoice-1B, a leading open-source audio model.",
    "Experience the power of MetaVoice-1B, an open-source audio model for text to speech conversion.",
    "MetaVoice-1B, a cutting-edge open-source audio model, brings text to life with speech synthesis.",
    "Harness the capabilities of MetaVoice-1B for seamless text to speech, an open-source audio model.",
    "Discover the efficiency of MetaVoice-1B in text to speech, an innovative open-source audio model.",
]


@app.websocket("/audio")
async def audio_stream(websocket: WebSocket):

    def flush():
        flush_queue(model.text_queue)
        flush_queue(model.embeddings_queue)
        flush_queue(model.audio_out_queue)

    start = time.perf_counter()
    await websocket.accept()
    flush()
    print(f"connection made with client: {time.perf_counter() - start}s, streaming...")

    try:
        while True:
            text = await websocket.receive_text()
            print(f"received: {text}")

            if text == "<client_end>":
                await websocket.send_text("<server_end>")
                continue

            # if text == "<flush>":
            #     flush()
            #     continue

            model.synthesise(
                text=text,
                spk_ref_path="https://cdn.themetavoice.xyz/speakers/bria.mp3",
                top_p=0.95,
                guidance_scale=3.0,
            )

            t0 = time.perf_counter()
            while True:
                try:
                    audio = model.audio_out_queue.get(timeout=0.5)
                    await websocket.send_bytes(audio.tobytes())
                except queue.Empty:
                    print(f"finished speaking this utterance in: {time.perf_counter() - t0}")
                    break

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e.code}, {e.reason}")

        os.kill(model.llm_process.pid, signal.SIGINT)
        os.kill(model.decoder_process.pid, signal.SIGINT)

        print("\n\n")

    except Exception as e:
        traceback.print_exc()
        print(f"error: {str(e)}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    model = TTS()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
