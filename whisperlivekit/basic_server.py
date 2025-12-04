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
    
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    logger.info(f"Saved upload to {tmp_path}")
    
    try:
        # Decode to PCM using FFmpeg
        logger.info("Decoding to PCM...")
        cmd = [
            "ffmpeg",
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
        logger.info(f"Decoded PCM data size: {len(pcm_data)} bytes")
        
        # Create AudioProcessor with PCM input forced
        # We need to hack this a bit since AudioProcessor reads from transcription_engine.args
        # We can subclass or just modify the instance
        
        audio_processor = AudioProcessor(
            transcription_engine=transcription_engine,
        )
        # Force PCM input mode
        audio_processor.is_pcm_input = True
        # Ensure ffmpeg_manager is not used/started
        audio_processor.ffmpeg_manager = None
        
        results_generator = await audio_processor.create_tasks()
        
        # Feed audio task
        async def feed_audio():
            chunk_size = 64 * 1024 
            total_size = len(pcm_data)
            processed = 0
            logger.info("Starting to feed PCM audio...")
            
            for i in range(0, total_size, chunk_size):
                chunk = pcm_data[i:i+chunk_size]
                await audio_processor.process_audio(chunk)
                processed += len(chunk)
                if i % (chunk_size * 100) == 0:
                     logger.info(f"Fed {processed}/{total_size} bytes")
                await asyncio.sleep(0)
                
            logger.info("Finished feeding audio. Signaling end of stream.")
            await audio_processor.process_audio(None)
            
        feed_task = asyncio.create_task(feed_audio())
        
        full_transcript = ""
        all_lines = []
        logger.info("Starting to consume results...")
        try:
            async for response in results_generator:
                logger.info(f"Response type: {type(response)}, Response: {response}")
                if hasattr(response, 'lines') and response.lines:
                     all_lines.extend(response.lines)
                     logger.info(f"Received {len(response.lines)} transcript lines")
                if hasattr(response, 'buffer_transcription') and response.buffer_transcription:
                     logger.info(f"Buffer transcription: {response.buffer_transcription}")
        except Exception as e:
            logger.error(f"Error in results loop: {e}", exc_info=True)
        
        await feed_task
        logger.info(f"Feed task done. Total lines collected: {len(all_lines)}")
        
        # Extract text from lines, skipping silent segments
        for line in all_lines:
            # Skip silent segments (speaker == -2)
            if hasattr(line, 'is_silence') and line.is_silence():
                continue
            if hasattr(line, 'speaker') and line.speaker == -2:
                continue
                
            if isinstance(line, str):
                full_transcript += line + " "
            elif hasattr(line, 'text') and line.text:
                full_transcript += line.text + " "
            else:
                logger.warning(f"Skipping line with no text: {line}")
        
        logger.info(f"Final transcript length: {len(full_transcript)}")
        return {"transcript": full_transcript.strip()}
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        raise e
    finally:
        logger.info("Cleaning up resources...")
        if 'audio_processor' in locals():
            await audio_processor.cleanup()
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
