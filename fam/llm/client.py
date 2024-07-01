import asyncio
import io
import time

import pyaudio
import websockets
from pydub import AudioSegment


async def receive_audio():
    p = pyaudio.PyAudio()
    stream = p.open(channels=1, rate=24000, output=True, format=pyaudio.paInt16)

    uri = "ws://localhost:8000/audio"

    start = time.perf_counter()
    time_to_first_byte = False
    async with websockets.connect(uri) as websocket:
        try:
            while True:
                data = await websocket.recv()
                if not time_to_first_byte:
                    print(f"first byte in: {time.perf_counter() - start}")
                    time_to_first_byte = True

                stream.write(data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


asyncio.run(receive_audio())
