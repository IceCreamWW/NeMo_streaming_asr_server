import logging
from nemo_live.client import Client

logging.basicConfig(level=logging.INFO)
client = Client("localhost", 9000)
client.transcribe_from_microphone()
# client.transcribe_from_audio_file("tests/jfk.flac")

