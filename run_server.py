import argparse

from nemo_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=9090, help="Websocket port to run the server on."
    )
    args = parser.parse_args()

    server = TranscriptionServer()
    server.run("0.0.0.0", port=args.port)
