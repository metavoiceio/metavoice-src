import os
import re
import subprocess
import tempfile

import librosa
import torch


def normalize_text(text: str) -> str:
    unicode_conversion = {
        8175: "'",
        8189: "'",
        8190: "'",
        8208: "-",
        8209: "-",
        8210: "-",
        8211: "-",
        8212: "-",
        8213: "-",
        8214: "||",
        8216: "'",
        8217: "'",
        8218: ",",
        8219: "`",
        8220: '"',
        8221: '"',
        8222: ",,",
        8223: '"',
        8228: ".",
        8229: "..",
        8230: "...",
        8242: "'",
        8243: '"',
        8245: "'",
        8246: '"',
        180: "'",
        2122: "TM",  # Trademark
    }

    text = text.translate(unicode_conversion)

    non_bpe_chars = set([c for c in list(text) if ord(c) >= 256])
    if len(non_bpe_chars) > 0:
        non_bpe_points = [(c, ord(c)) for c in non_bpe_chars]
        raise ValueError(f"Non-supported character found: {non_bpe_points}")

    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("*", " ")
    text = text.strip()
    text = re.sub("\s\s+", " ", text)  # remove multiple spaces
    return text


def check_audio_file(path_or_uri, threshold_s=30):
    if "http" in path_or_uri:
        temp_fd, filepath = tempfile.mkstemp()
        os.close(temp_fd)  # Close the file descriptor, curl will create a new connection
        curl_command = ["curl", "-L", path_or_uri, "-o", filepath]
        subprocess.run(curl_command, check=True)

    else:
        filepath = path_or_uri

    audio, sr = librosa.load(filepath)
    duration_s = librosa.get_duration(y=audio, sr=sr)
    if duration_s < threshold_s:
        raise Exception(
            f"The audio file is too short. Please provide an audio file that is at least {threshold_s} seconds long to proceed."
        )

    # Clean up the temporary file if it was created
    if "http" in path_or_uri:
        os.remove(filepath)


def get_default_dtype() -> str:
    """Compute default 'dtype' based on GPU architecture"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            dtype = "float16" if device_properties.major <= 7 else "bfloat16"  # tesla and turing architectures
    else:
        dtype = "float16"

    print(f"using dtype={dtype}")
    return dtype


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
