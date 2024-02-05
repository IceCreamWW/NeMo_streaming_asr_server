FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /workspace

RUN chmod 1777 /tmp && apt update && apt install git build-essential portaudio19-dev -y && git config --global http.proxy http://127.0.0.1:7895
COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir && python -m pip install git+https://github.com/NVIDIA/NeMo.git@main\#egg=nemo_toolkit[all]

# -v /mnt/disk2/home/vv/downloads/cache/:/mnt/nas/cache
ENV XDG_CACHE_HOME=/mnt/nas/cache
ENV NEMO_CACHE_DIR=/mnt/nas/cache/torch/NeMo

COPY nemo_live nemo_live
COPY run_server.py .

EXPOSE 9000
CMD [ "python3", "run_server.py", "--port", "9000"]

