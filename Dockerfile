FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as base

# Install system dependencies in a single RUN command to reduce layers
# Combine apt-get update, upgrade, and installation of packages. Clean up in the same layer to reduce image size.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip git wget curl build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz &&\
    wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5 &&\
    md5sum -c ffmpeg-git-amd64-static.tar.xz.md5 &&\
    tar xvf ffmpeg-git-amd64-static.tar.xz &&\
    mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/ &&\
    rm -rf ffmpeg-git-*

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir packaging wheel torch
RUN pip install --no-cache-dir flash-attn
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python3.10", "fam/llm/serving.py"]