import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from whisperlivekit import (AudioProcessor, TranscriptionEngine,
                            get_inline_ui_html, parse_args)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):    
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global transcription_engine
    logger.info(f"Received upload request: {file.filename}")
    
    import tempfile
    import os
    import numpy as np
    
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    logger.info(f"Saved upload to {tmp_path}")
    
    try:
        # Decode to PCM using FFmpeg
        logger.info(f"Decoding audio file...")
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", tmp_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {stderr.decode()}")
            return {"error": "FFmpeg decoding failed"}
            
        pcm_data = stdout
        logger.info(f"Decoded PCM data: {len(pcm_data)} bytes")
        
        # Convert PCM bytes to float32 numpy array
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        logger.info(f"Converted to audio array: shape={audio.shape}, dtype={audio.dtype}")
        
        # Use direct Whisper transcription (works better for translation than streaming)
        logger.info("Starting transcription...")
        from whisperlivekit.whisper import transcribe
        
        # Get transcription options from the engine
        task = "translate" if transcription_engine.args.direct_english_translation else "transcribe"
        
        logger.info(f"DEBUG: direct_english_translation={transcription_engine.args.direct_english_translation}")
        logger.info(f"Starting language detection and {task}...")
        
        # For translation to work reliably, we need to detect language first
        # then pass it explicitly to the transcribe function
        if task == "translate":
            # Detect language first
            from whisperlivekit.whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
            import torch
            
            mel = log_mel_spectrogram(audio, transcription_engine.asr.model.dims.n_mels)
            mel_segment = pad_or_trim(mel, N_FRAMES).to(transcription_engine.asr.model.device)
            _, probs = transcription_engine.asr.model.detect_language(mel_segment)
            detected_language = max(probs, key=probs.get)
            
            logger.info(f"Detected language: {detected_language}, translating to English")
            language = detected_language
        else:
            # For transcription, auto-detect is fine
            language = None
            logger.info(f"Transcribing with language=auto-detect")
        
        # Run transcription in thread pool to avoid blocking
        result = await asyncio.to_thread(
            transcribe,
            transcription_engine.asr.model,
            audio,
            language=language,
            task=task,
            word_timestamps=True,
            verbose=False,
            temperature=0.0,  # Deterministic output, no randomness
            beam_size=5,  # Use beam search for better quality
            best_of=5,  # Try 5 candidates and pick the best
        )
        
        full_transcript = result.get("text", "").strip()
        logger.info(f"Transcription complete: {len(full_transcript)} characters")
        logger.info(f"Detected language: {result.get('language', 'N/A')}")
        logger.info(f"Number of segments: {len(result.get('segments', []))}")
        
        return {"transcript": full_transcript}
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Cleaning up resources...")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.info("Cleanup done.")


@app.websocket("/chunk")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = { **uvicorn_kwargs, "forwarded_allow_ips" : args.forwarded_allow_ips }

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
