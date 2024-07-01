import queue
from fastapi import FastAPI, WebSocket
import asyncio
import random
from starlette.websockets import WebSocketDisconnect
import traceback
import torch
import time
from fam.llm.fast_inference import TTS

app = FastAPI()
model = None
texts = [
    "This is an example of speech synthesis using MetaVoice-1B, a leading open-source audio model.",
    "Experience the power of MetaVoice-1B, an open-source audio model for text to speech conversion.",
    "MetaVoice-1B, a cutting-edge open-source audio model, brings text to life with speech synthesis.",
    "Harness the capabilities of MetaVoice-1B for seamless text to speech, an open-source audio model.",
    "Discover the efficiency of MetaVoice-1B in text to speech, an innovative open-source audio model."
]


def flush_queue(kqueue):
    while not kqueue.empty():
        try:
            kqueue.get_nowait()
        except queue.Empty:
            pass


@app.websocket("/audio")
async def audio_stream(websocket: WebSocket):
    # text = random.choice(texts)

    await websocket.accept()
    print("connection made with client, streaming...")    

    flush_queue(model.audio_out_queue)
    try:
        while True:
            text = await websocket.receive_text()
            print(f"received: {text}")
                    
            if text == "<client_end>":
                await websocket.send_text("<server_end>")
                flush_queue(model.audio_out_queue)
                print("\n\n")
                continue

            model.synthesise(
                text=text,
                spk_ref_path="https://cdn.themetavoice.xyz/speakers/bria.mp3",
                top_p=0.95,
                guidance_scale=3.0,
            )

            while True:
                try:
                    # t0 = time.perf_counter()
                    audio = model.audio_out_queue.get(timeout=2)
                    await websocket.send_bytes(audio.tobytes())
                    # print(f"got in: {time.perf_counter() - t0}")
                
                except queue.Empty:
                    print("finished speaking this utterance")
                    break
    
    except WebSocketDisconnect as e:
        traceback.print_exc()
        print(f"WebSocket disconnected: {e.code}, {e.reason}")
    except Exception as e:
        traceback.print_exc()
        print(f"error: {str(e)}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    model = TTS()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)