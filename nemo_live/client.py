import librosa
import logging
import pyaudio

import numpy as np
import threading
import json
import websocket
import uuid
import time

class Client:
    def __init__(self, host, port):
        self.chunk = 8000
        self.recording = False
        self.uid = str(uuid.uuid4())
        self.last_response_recieved = None
        self.server_timeout_seconds = 10

        socket_url = f"ws://{host}:{port}"
        self.client_socket = websocket.WebSocketApp(
            socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=lambda ws, error: logging.error(f"Error: {error}"),
        )

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        logging.info("connecting to server")
        while not self.recording:
            time.sleep(.5)
        logging.info("connected to server")

    def on_message(self, ws, message):
        self.last_response_recieved = time.time()
        message = json.loads(message)

        status = message.get("status", "")
        msg = message.get("message", "")

        if status != "OK":
            logging.error(f"Server {status}: {msg}")
            self.recording = False
            return

        self.recording = True
        text = message.get("text", "")

        print(f"text: {text}")

    def on_open(self, ws):
        ws.send(json.dumps({"uid": self.uid}))

    def send_packet_to_server(self, message):
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    def close_websocket(self):
        self.client_socket.close()
        self.ws_thread.join()

    def transcribe_from_audio_file(self, filename, simulate_streaming=True):
        audio, sr = librosa.load(filename, sr=16000)
        audio = (audio * 32768).astype(np.int16)
        if simulate_streaming:
            for i in range(0, len(audio), self.chunk):
                assert self.recording
                audio_chunk = audio[i:i+self.chunk]
                self.send_packet_to_server(audio_chunk.tobytes())
                time.sleep(self.chunk / sr)
        else:
            self.send_packet_to_server(audio.tobytes())

        assert self.last_response_recieved
        while time.time() - self.last_response_recieved < self.server_timeout_seconds:
            time.sleep(1)
            continue
        self.close_websocket()

    def transcribe_from_microphone(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk,
        )
        try:
            while self.recording:
                data = stream.read(self.chunk, exception_on_overflow = False)
                data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.send_packet_to_server(data.tobytes())
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.close_websocket()

