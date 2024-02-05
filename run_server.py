import argparse

from nemo_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=9000) 
    args = parser.parse_args()

    server = TranscriptionServer()
    server.run("0.0.0.0", port=args.port)
