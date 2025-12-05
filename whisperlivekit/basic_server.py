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
        seen_segments = set()  # Track unique segments by (start, end, text)
        last_response = None
        logger.info("Starting to consume results...")
        
        # We need to run feeding and consuming concurrently
        # and ensure feeding completes so that it triggers the "None" signal
        # which eventually terminates the results stream.
        
        async def consume_results():
            nonlocal last_response
            async for response in results_generator:
                if hasattr(response, 'lines') and response.lines:
                     last_response = response  
                     logger.info(f"Received {len(response.lines)} lines")
                     for line in response.lines:
                         # Debug logging
                         if hasattr(line, 'translation'):
                             logger.info(f"DEBUG: Line translation: {line.translation}")
                         if hasattr(line, 'text'):
                             logger.info(f"DEBUG: Line text: {line.text}")

        # Run both tasks
        # feed_task will finish when all audio is sent + None sentinel is sent.
        # consume_results will finish when generator exhausts (after None sentinel triggers stop).
        await asyncio.gather(feed_task, consume_results())
        
        # feed_task is awaited in gather above
        
        # Give a small buffer time for final processing after feed_audio sends None
        # We need to make sure we consume the generator until it's actually done.
        # The loop "async for response in results_generator" should naturally finish when generator closes.
        # But feed_task finishes when it puts None into queue. 
        # The generator might still yield a few results after that.
        # The current code structure "await feed_task" happens *inside* the "await asyncio.gather" 
        # structure usually, but here it is separate.
        # Wait, the code has "async for ...:". This blocks until generator is done.
        # feed_task runs in background.
        # So "await feed_task" here is actually unreachable until the loop finishes!
        # AND the loop finishes only when generator finishes.
        # Generator finishes when audio_processor processes None.
        
        # Ah! The issue is:
        # We start loop `async for response`.
        # We start feed_task.
        # feed_task feeds audio, then feeds None.
        # audio_processor sees None, sets is_stopping, puts SENTINEL in queues.
        # Processors finish.
        # results_formatter sees stopping flag, finishes.
        # Generator stops yielding.
        # Loop finishes.
        # THEN we await feed_task (which is already done).
        
        # So the flow is correct.
        # However, "force final processing" might be needed if vac/buffers hold data.
        # The audio_processor.process_audio(None) triggers expected shutdown.
        
        # But let's ensuring we capture the VERY LAST response which might contain the final buffered text.
        pass
        
        # Use only the final response which has the complete transcript
        final_lines = []
        if last_response and hasattr(last_response, 'lines'):
            final_lines = last_response.lines
        
        logger.info(f"Total final lines: {len(final_lines)}")
        
        # Extract text from lines, skipping silent segments and duplicates
        for line in final_lines:
            # Skip silent segments (speaker == -2)
            if hasattr(line, 'is_silence') and callable(line.is_silence) and line.is_silence():
                continue
            if hasattr(line, 'speaker') and line.speaker == -2:
                continue
            
            # Create unique key for this segment
            if hasattr(line, 'start') and hasattr(line, 'end') and hasattr(line, 'text'):
                segment_key = (line.start, line.end, line.text)
                if segment_key in seen_segments:
                    continue
                seen_segments.add(segment_key)
                
            # Check for translation first
            text_part = ""
            if hasattr(line, 'translation') and line.translation:
                # translation might be a string or object depending on implementation
                # Based on tokens_alignment fix, it's a string.
                text_val = line.translation
                if hasattr(text_val, 'text'): # safeguard if it remains an object
                    text_part = text_val.text
                else:
                    text_part = str(text_val)
            
            # Fallback to original text if no translation
            if not text_part and hasattr(line, 'text') and line.text:
                text_part = line.text
                
            if text_part:
                full_transcript += text_part + " "
        
        logger.info(f"Final transcript: {full_transcript[:100]}...")
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
