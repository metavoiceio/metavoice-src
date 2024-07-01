import hashlib
import json
import os
import re
import subprocess
import tempfile
import pathlib
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

    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace("*", " ").strip()
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
    # if duration_s < threshold_s:
    #     raise Exception(
    #         f"The audio file is too short. Please provide an audio file that is at least {threshold_s} seconds long to proceed."
    #     )

    # Clean up the temporary file if it was created
    if "http" in path_or_uri:
        os.remove(filepath)


def get_default_dtype() -> str:
    """Compute default 'dtype' based on GPU architecture"""
    return "bfloat16"
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


def hash_dictionary(d: dict):
    # Serialize the dictionary into JSON with sorted keys to ensure consistency
    serialized = json.dumps(d, sort_keys=True)
    # Encode the serialized string to bytes
    encoded = serialized.encode()
    # Create a hash object (you can also use sha1, sha512, etc.)
    hash_object = hashlib.sha256(encoded)
    # Get the hexadecimal digest of the hash
    hash_digest = hash_object.hexdigest()
    return hash_digest


def get_cached_embedding(local_file_path: str, spkemb_model):
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File {local_file_path} not found!")

    # hash the file path to get the cache name
    _cache_name = "embedding_" + hashlib.md5(local_file_path.encode("utf-8")).hexdigest() + ".pt"

    os.makedirs(os.path.expanduser("~/.cache/fam/"), exist_ok=True)
    cache_path = os.path.expanduser(f"~/.cache/fam/{_cache_name}")

    if not os.path.exists(cache_path):
        spk_emb = spkemb_model.embed_utterance_from_file(local_file_path, numpy=False).unsqueeze(0)  # (b=1, c)
        torch.save(spk_emb, cache_path)
    else:
        spk_emb = torch.load(cache_path)

    return spk_emb


def get_cached_file(file_or_uri: str):
    """
    If it's an s3 file, download it to a local temporary file and return that path.
    Otherwise return the path as is.
    """
    is_uri = file_or_uri.startswith("http")

    cache_path = None
    if is_uri:
        ext = pathlib.Path(file_or_uri).suffix
        # hash the file path to get the cache name
        _cache_name = "audio_" + hashlib.md5(file_or_uri.encode("utf-8")).hexdigest() + ext

        os.makedirs(os.path.expanduser("~/.cache/metavoice/"), exist_ok=True)
        cache_path = os.path.expanduser(f"~/.cache/metavoice/{_cache_name}")

        if not os.path.exists(cache_path):
            command = f"curl -o {cache_path} {file_or_uri}"
            subprocess.run(command, shell=True, check=True)
    else:
        if os.path.exists(file_or_uri):
            cache_path = file_or_uri
        else:
            raise FileNotFoundError(f"File {file_or_uri} not found!")
    return cache_path