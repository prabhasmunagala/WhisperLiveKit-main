import argparse
import asyncio
import aiohttp
import websockets
import json
import os

async def upload_file(url, file_path):
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as f:
            data = {'file': f}
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Transcript: {result.get('transcript')}")
                else:
                    print(f"Error: {response.status}")
                    print(await response.text())

async def stream_file(url, file_path):
    async with websockets.connect(url) as websocket:
        # Wait for config message
        message = await websocket.recv()
        config = json.loads(message)
        print(f"Server config: {config}")

        # Send audio in chunks
        chunk_size = 4096 
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                await websocket.send(data)
                # Small delay to simulate real-time streaming (optional, but good for testing)
                await asyncio.sleep(0.01) 
        
        print("Finished sending audio. Waiting for final results...")
        # Send empty bytes to signal end of stream (if server expects it, though connection close might be enough)
        # Based on server code: if 'bytes' in str(e) -> Client has closed connection.
        # But server also handles empty message as stop signal in process_audio.
        # However, websocket.send(b'') might not be valid or necessary if we just close.
        # Let's keep listening until server sends "ready_to_stop"
        
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                if data.get("type") == "ready_to_stop":
                    print("Server signaled ready to stop.")
                    break
                
                if "lines" in data and data["lines"]:
                    print(f"Partial Transcript: {data['lines']}")
                if "buffer_transcription" in data and data["buffer_transcription"]:
                     print(f"Buffer: {data['buffer_transcription']}", end='\r')

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")

def main():
    parser = argparse.ArgumentParser(description="WhisperLiveKit Client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a .wav file for transcription")
    upload_parser.add_argument("file_path", help="Path to the .wav file")
    upload_parser.add_argument("--url", default="http://localhost:8000/upload", help="Server URL")

    stream_parser = subparsers.add_parser("stream", help="Stream a .wav file for transcription")
    stream_parser.add_argument("file_path", help="Path to the .wav file")
    stream_parser.add_argument("--url", default="ws://localhost:8000/chunk", help="Server WebSocket URL")

    args = parser.parse_args()

    if args.command == "upload":
        asyncio.run(upload_file(args.url, args.file_path))
    elif args.command == "stream":
        asyncio.run(stream_file(args.url, args.file_path))

if __name__ == "__main__":
    main()
