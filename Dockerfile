FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as base

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single RUN command to reduce layers
# Combine apt-get update, upgrade, and installation of packages. Clean up in the same layer to reduce image size.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip git wget curl build-essential pipx && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install via pip given ubuntu 22.04 as per docs https://pipx.pypa.io/stable/installation/
RUN python3 -m pip install --user pipx && \
    python3 -m pipx ensurepath && \
    python3 -m pipx install poetry==1.8.2

# make pipx installs (i.e poetry) available
ENV PATH="/root/.local/bin:${PATH}"

# install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz &&\
    wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5 &&\
    md5sum -c ffmpeg-git-amd64-static.tar.xz.md5 &&\
    tar xvf ffmpeg-git-amd64-static.tar.xz &&\
    mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/ &&\
    rm -rf ffmpeg-git-*

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md # poetry will complain otherwise

RUN poetry install --without dev --no-root
RUN poetry run python -m pip install torch==2.2.1 torchaudio==2.2.1 && \
  rm -rf $POETRY_CACHE_DIR

COPY fam ./fam
COPY serving.py ./
COPY app.py ./

RUN poetry install --only-root

ENTRYPOINT ["poetry", "run", "python", "serving.py"]
