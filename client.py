import argparse
import asyncio
import time
import aiohttp
import websockets
import json
import os

async def upload_file(url, file_path, output_file=None):
    print(f"Uploading {file_path} to {url}...")
    print("Processing... this may take a while for large files.")
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as f:
            data = {'file': f}
            # Increase timeout for long files
            timeout = aiohttp.ClientTimeout(total=3600) 
            async with session.post(url, data=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    transcript = result.get('transcript')
                    diarization = result.get('diarization')
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Processing complete in {elapsed_time:.2f} seconds.")
                    
                    if output_file:
                        with open(output_file, 'w') as f_out:
                            if output_file.endswith('.json'):
                                json.dump(result, f_out, indent=2)
                            else:
                                f_out.write(transcript)
                        print(f"Output saved to {output_file}")
                    else:
                        print(f"\n{'='*60}")
                        print("TRANSCRIPT:")
                        print(f"{'='*60}")
                        print(transcript)
                        
                        # Print diarization results if available
                        if diarization:
                            print(f"\n{'='*60}")
                            print("CONVERSATION (Speaker Diarization):")
                            print(f"{'='*60}")
                            for seg in diarization:
                                text = seg.get('text', '')
                                if text:
                                    print(f"\n[{seg['speaker']}] ({seg['start']:.1f}s - {seg['end']:.1f}s)")
                                    print(f"  \"{text}\"")
                                else:
                                    print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
                            print(f"\n{'='*60}")
                            print(f"Total speaker turns: {len(diarization)}")
                else:
                    print(f"Error: {response.status}")
                    print(await response.text())

async def stream_file(url, file_path):
    """Stream a file to the WebSocket server for real-time transcription."""
    import os
    
    file_size = os.path.getsize(file_path)
    print(f"Streaming {file_path} ({file_size / 1024:.1f} KB) to {url}...")
    start_time = time.time()
    
    async with websockets.connect(url, ping_interval=30, ping_timeout=60) as websocket:
        # Wait for config message
        message = await websocket.recv()
        config = json.loads(message)
        print(f"Server config: {config}")
        use_pcm = config.get("useAudioWorklet", False)
        print(f"Mode: {'PCM' if use_pcm else 'Encoded audio'}")
        print("-" * 60)

        # Send audio in chunks
        chunk_size = 4096  # 4KB chunks
        bytes_sent = 0
        
        async def send_audio():
            nonlocal bytes_sent
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    await websocket.send(data)
                    bytes_sent += len(data)
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.01)
            
            print(f"\n[INFO] Finished sending audio ({bytes_sent / 1024:.1f} KB)")
            print("[INFO] Waiting for final results...")
            # Send empty message to signal end of stream
            try:
                await websocket.send(b'')
            except Exception as e:
                print(f"[DEBUG] Could not send empty signal: {e}")

        async def receive_results():
            last_lines = ""
            last_buffer = ""
            
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=120)
                    data = json.loads(response)
                    
                    if data.get("type") == "ready_to_stop":
                        elapsed = time.time() - start_time
                        print(f"\n{'=' * 60}")
                        print(f"[DONE] Streaming complete in {elapsed:.2f} seconds")
                        return
                    
                    if data.get("status") == "error":
                        print(f"\n[ERROR] {data.get('error', 'Unknown error')}")
                        continue
                    
                    # Display transcription updates
                    lines = data.get("lines", "")
                    buffer = data.get("buffer_transcription", "")
                    
                    if lines and lines != last_lines:
                        print(f"\n[TRANSCRIPT] {lines}")
                        last_lines = lines
                    
                    if buffer and buffer != last_buffer:
                        # Show buffer on same line (partial results)
                        print(f"  [partial] {buffer}", end='\r')
                        last_buffer = buffer
                    
                    # Show remaining processing time if available
                    remaining = data.get("remaining_time_transcription", 0)
                    if remaining > 0:
                        print(f"  [processing lag: {remaining:.1f}s]", end='\r')

            except asyncio.TimeoutError:
                print("\n[TIMEOUT] No response from server for 120 seconds")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n[INFO] Connection closed: {e}")

        # Run send and receive concurrently
        await asyncio.gather(
            send_audio(),
            receive_results(),
            return_exceptions=True
        )

async def stream_microphone(url, device_index=None):
    """Stream audio from microphone to the WebSocket server for real-time transcription."""
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pyaudio is required for microphone streaming.")
        print("Install with: pip install pyaudio")
        return

    # Audio settings (must match server expectations)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024  # Samples per chunk
    FORMAT = pyaudio.paInt16
    
    print(f"🎤 Connecting to {url}...")
    print("=" * 60)
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # List available input devices if requested
    if device_index is None:
        print("Available audio input devices:")
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"  [{i}] {dev_info['name']}")
        print("-" * 60)
    
    try:
        # Open microphone stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"✅ Microphone opened (Rate: {SAMPLE_RATE}Hz, Channels: {CHANNELS})")
        print("🎙️  Speak now... Press Ctrl+C to stop.\n")
        print("-" * 60)
        
    except Exception as e:
        print(f"ERROR: Could not open microphone: {e}")
        audio.terminate()
        return

    start_time = time.time()
    stop_event = asyncio.Event()
    
    async with websockets.connect(url, ping_interval=30, ping_timeout=60) as websocket:
        # Wait for config message
        message = await websocket.recv()
        config = json.loads(message)
        print(f"Server config: {config}")
        
        async def send_audio():
            """Continuously capture and send microphone audio."""
            try:
                while not stop_event.is_set():
                    try:
                        # Read audio data (non-blocking with exception_on_overflow=False)
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        await websocket.send(data)
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    except IOError as e:
                        # Handle buffer overflow gracefully
                        print(f"[WARN] Audio buffer overflow: {e}")
                        continue
            except asyncio.CancelledError:
                pass
            finally:
                print("\n[INFO] Stopping microphone capture...")
                stream.stop_stream()
                stream.close()
                audio.terminate()
                # Signal end of stream
                try:
                    await websocket.send(b'')
                except:
                    pass

        async def receive_results():
            """Receive and display transcription results."""
            last_text = ""
            full_transcript = []
            
            try:
                while not stop_event.is_set():
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        data = json.loads(response)
                        
                        if data.get("type") == "ready_to_stop":
                            elapsed = time.time() - start_time
                            print(f"\n{'=' * 60}")
                            print(f"[DONE] Session complete in {elapsed:.2f} seconds")
                            stop_event.set()
                            return
                        
                        if data.get("status") == "error":
                            print(f"\n[ERROR] {data.get('error', 'Unknown error')}")
                            continue
                        
                        # Get current elapsed time
                        elapsed = time.time() - start_time
                        timestamp = f"[{elapsed:.1f}s]"
                        
                        # Extract text from lines (list of segment dictionaries)
                        lines = data.get("lines", [])
                        buffer = data.get("buffer_transcription", "")
                        
                        # Extract text from line segments
                        text_parts = []
                        for line in lines:
                            if isinstance(line, dict):
                                text = line.get("text", "").strip()
                                if text:
                                    text_parts.append(text)
                            elif isinstance(line, str):
                                text_parts.append(line)
                        
                        current_text = " ".join(text_parts)
                        
                        if current_text and current_text != last_text:
                            # Clear the current line and print confirmed text with timestamp
                            print(f"\r{'':100}")  # Clear line
                            print(f"\033[92m✓ {timestamp} {current_text}\033[0m")  # Green text for confirmed
                            full_transcript.append(f"{timestamp} {current_text}")
                            last_text = current_text
                        
                        if buffer:
                            # Show partial results with timestamp (in yellow)
                            print(f"\r\033[93m⏳ {timestamp} {buffer}\033[0m", end='', flush=True)
                        
                    except asyncio.TimeoutError:
                        # Timeout is normal during silence, just continue
                        continue
                        
            except asyncio.CancelledError:
                pass
            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n[INFO] Connection closed: {e}")
            finally:
                if full_transcript:
                    print(f"\n{'=' * 60}")
                    print("FULL TRANSCRIPT:")
                    print("-" * 60)
                    print(" ".join(full_transcript))
                    print("=" * 60)

        # Handle Ctrl+C gracefully
        async def wait_for_interrupt():
            """Wait for keyboard interrupt."""
            try:
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                stop_event.set()

        try:
            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                receive_results(),
                wait_for_interrupt(),
                return_exceptions=True
            )
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="WhisperLiveKit Client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a .wav file for transcription")
    upload_parser.add_argument("file_path", help="Path to the .wav file")
    upload_parser.add_argument("--url", default="http://localhost:8000/upload", help="Server URL")
    upload_parser.add_argument("--output", help="Path to save the output (e.g., result.json or result.txt)")

    stream_parser = subparsers.add_parser("stream", help="Stream a .wav file for transcription")
    stream_parser.add_argument("file_path", help="Path to the .wav file")
    stream_parser.add_argument("--url", default="ws://localhost:8000/chunk", help="Server WebSocket URL")

    mic_parser = subparsers.add_parser("mic", help="Stream from microphone for real-time transcription")
    mic_parser.add_argument("--url", default="ws://localhost:8000/chunk", help="Server WebSocket URL")
    mic_parser.add_argument("--device", type=int, default=None, help="Audio input device index (optional)")

    args = parser.parse_args()

    if args.command == "upload":
        asyncio.run(upload_file(args.url, args.file_path, args.output))
    elif args.command == "stream":
        asyncio.run(stream_file(args.url, args.file_path))
    elif args.command == "mic":
        asyncio.run(stream_microphone(args.url, args.device))

if __name__ == "__main__":
    main()
