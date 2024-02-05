import logging
from nemo_live.client import Client

logging.basicConfig(level=logging.INFO)
client = Client("localhost", 9001)
client.transcribe_from_audio_file("/mnt/wsl/PHYSICALDRIVE0p3/home/vv/downloads/data/zheda240130/audios/1_en.wav")
# client.transcribe_from_audio_file("tests/jfk.flac")

