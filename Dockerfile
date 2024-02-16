FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as base
RUN apt-get update && \  
    apt-get upgrade -y 

RUN apt-get update && \  
    # apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip git wget curl && \  
    apt-get autoremove -y && \  
    apt-get clean

# install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz &&\
    wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5 &&\
    md5sum -c ffmpeg-git-amd64-static.tar.xz.md5 &&\
    tar xvf ffmpeg-git-amd64-static.tar.xz &&\
    mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/ &&\
    rm -rf ffmpeg-git-*

RUN pip install torch packaging wheel &&\
    pip install flash-attn
WORKDIR /app
COPY requirements.txt requirements.txt
# install python packages
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
RUN apt-get remove -y build-essential && \  
    apt-get autoremove -y && \  
    apt-get clean  && \  
    rm -rf /var/lib/apt/lists/*  && \
    pip cache purge

ENTRYPOINT [ "python3.10", "fam/llm/serving.py" ]