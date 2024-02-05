import json
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from websockets.sync.server import serve

from nemo_live.transcriber import NeMoTranscriber as Transcriber


class TranscriptionServer:
    """
    Represents a transcription server that handles incoming audio from clients.

    Attributes:
        RATE (int): The audio sampling rate (constant) set to 16000.
        clients (dict): A dictionary to store connected clients.
        websockets (dict): A dictionary to store WebSocket connections.
        clients_start_time (dict): A dictionary to track client start times.
        max_clients (int): Maximum allowed connected clients.
        max_connection_time (int): Maximum allowed connection time in seconds.
    """

    RATE = 16000

    def __init__(self):
        self.clients = {}
        self.websockets = {}
        self.clients_start_time = {}
        self.max_clients = 4

    def recv_audio(self, websocket):
        """
        Args:
            websocket (WebSocket): The WebSocket connection for the client.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        logging.info("New client connected")
        options = websocket.recv()
        options = json.loads(options)

        if len(self.clients) >= self.max_clients:
            logging.warning("Client Queue Full. Asking client to wait ...")
            response = {
                "uid": options["uid"],
                "status": "ERROR",
                "message": f"Reached maximum number of clients: {self.max_clients}",
            }
            websocket.send(json.dumps(response))
            websocket.close()
            del websocket
            return

        client = ServeClient(websocket, client_uid=options["uid"])
        self.clients[websocket] = client
        self.clients_start_time[websocket] = time.time()

        while True:
            try:
                frame_data = websocket.recv()
                frame_np = (np.frombuffer(frame_data, dtype=np.int16) / 32768.0).astype(np.float32)
                self.clients[websocket].add_frames(frame_np)
            except Exception as e:
                logging.error(e)
                self.clients[websocket].cleanup()
                self.clients.pop(websocket)
                self.clients_start_time.pop(websocket)
                del websocket
                break

    def run(self, host, port=9090):
        with serve(self.recv_audio, host, port) as server:
            server.serve_forever()


class ServeClient(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(self, websocket, client_uid):
        self.websocket = websocket
        self.client_uid = client_uid
        self.frames_np = None
        self.exit = False

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "status": "OK",
                    # "message": self.SERVER_READY,
                }
            )
        )

        # threading
        self.lock = threading.Lock()

    def add_frames(self, frame_np):
        """
        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array; dtype: np.float32
        """
        with self.lock:
            if self.frames_np is None:
                self.frames_np = frame_np
            else:
                self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def speech_to_text(self):
        states = None
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None or self.frames_np.shape[0] < Transcriber.chunk_size_samples:
                time.sleep(0.1)
                continue

            chunk = self.frames_np[: Transcriber.chunk_size_samples]
            with self.lock:
                self.frames_np = self.frames_np[Transcriber.chunk_size_samples :]

            text, states = Transcriber.transcribe_chunk(chunk, states)
            if not len(text):
                continue
            try:
                self.websocket.send(
                    json.dumps({"uid": self.client_uid, "status": "OK", "text": text})
                )
            except Exception as e:
                logging.error(f"[ERROR]: Failed to send message to client: {e}")

    def disconnect(self):
        self.websocket.send(json.dumps({"uid": self.client_uid, "status": self.DISCONNECT,}))

    def cleanup(self):
        self.exit = True

