import asyncio
import gc
import logging
from contextlib import asynccontextmanager

import torch
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
    
    # Cleanup on shutdown
    logger.info("Shutting down server - cleaning up memory...")
    
    # Clear the transcription engine
    if transcription_engine is not None:
        # Clear ASR model if exists
        if hasattr(transcription_engine, 'asr') and transcription_engine.asr is not None:
            if hasattr(transcription_engine.asr, 'model'):
                del transcription_engine.asr.model
            if hasattr(transcription_engine.asr, 'pipe'):
                del transcription_engine.asr.pipe
        transcription_engine = None
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared.")
    
    logger.info("Memory cleanup complete.")

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
        
        # Add 2 seconds of silence padding to ensure the last segment is fully transcribed
        # Whisper often cuts off the final phrase without sufficient trailing silence
        padding = np.zeros(int(16000 * 2.0), dtype=np.float32)
        audio = np.concatenate((audio, padding))
        
        logger.info(f"Converted to audio array: shape={audio.shape}, dtype={audio.dtype}")
        
        # Use direct Whisper transcription (works better for translation than streaming)
        logger.info("Starting transcription...")
        word_timestamps = []  # Store word-level timestamps for diarization alignment
        
        if transcription_engine.asr.backend_choice == "transformers":
            # For transformers backend, get word-level timestamps for diarization
            result = await asyncio.to_thread(
                transcription_engine.asr.transcribe,
                audio,
                return_timestamps="word"  # Get word-level timestamps
            )
            # TRANSFORMERS BACKEND: result is already a dict from `pipeline`
            full_transcript = result.get("text", "").strip()
            
            # Extract word timestamps from chunks
            chunks = result.get("chunks", [])
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")
                if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                    start, end = timestamp
                    if start is not None and end is not None and text:
                        word_timestamps.append({
                            "word": text,
                            "start": start,
                            "end": end
                        })
            
        else:
            # Fallback for other backends (Whisper, Faster-Whisper, etc) 
            # Existing logic for non-transformers backends
            
            # Detect language first if translating
            language = None
            if task == "translate":
                # Only attempt manual detection if the model has the standard Whisper structure
                if hasattr(transcription_engine.asr, "model") and hasattr(transcription_engine.asr.model, "dims"):
                    from whisperlivekit.whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
                    mel = log_mel_spectrogram(audio, transcription_engine.asr.model.dims.n_mels)
                    mel_segment = pad_or_trim(mel, N_FRAMES).to(transcription_engine.asr.model.device)
                    _, probs = transcription_engine.asr.model.detect_language(mel_segment)
                    detected_language = max(probs, key=probs.get)
                    language = detected_language
                    logger.info(f"Detected language: {detected_language}, translating to English")
                else:
                    logger.info("Skipping explicit language detection (model structure varies). Letting backend handle it.")

            from whisperlivekit.whisper import transcribe
            result = await asyncio.to_thread(
                transcribe,
                transcription_engine.asr.model,
                audio,
                language=language,
                task=task,
                word_timestamps=True,
                verbose=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
            )
            full_transcript = result.get("text", "").strip()
            
            # Extract word timestamps from segments
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_timestamps.append({
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0),
                        "end": word_info.get("end", 0)
                    })
        

        logger.info(f"Transcription complete: {len(full_transcript)} characters")
        logger.info(f"Detected language: {result.get('language', 'N/A')}")
        logger.info(f"Number of word timestamps: {len(word_timestamps)}")
        
        # Diarization processing
        diarization_result = None
        if args.diarization:
            logger.info("Starting diarization with Sortformer...")
            try:
                from whisperlivekit.diarization.sortformer_backend import (
                    SortformerDiarization,
                    SortformerDiarizationOnline,
                )
                
                # Initialize Sortformer diarization
                if not hasattr(transcription_engine, '_diar_model') or transcription_engine._diar_model is None:
                    logger.info("Loading Sortformer diarization model...")
                    transcription_engine._diar_model = SortformerDiarization()
                
                diar_online = SortformerDiarizationOnline(
                    shared_model=transcription_engine._diar_model,
                    sample_rate=16000
                )
                
                # Process audio in chunks for diarization
                chunk_size = int(16000 * 1.0)  # 1 second chunks
                all_segments = []
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    diar_online.insert_audio_chunk(chunk)
                    segments = await diar_online.diarize()
                    if segments:
                        all_segments.extend(segments)
                
                # Close diarization
                diar_online.close()
                
                # Merge consecutive segments from the same speaker
                if all_segments:
                    merged_segments = []
                    current_speaker = None
                    current_start = None
                    current_end = None
                    
                    for seg in all_segments:
                        if current_speaker is None:
                            # First segment
                            current_speaker = seg.speaker
                            current_start = seg.start
                            current_end = seg.end
                        elif seg.speaker == current_speaker:
                            # Same speaker, extend the segment
                            current_end = seg.end
                        else:
                            # Different speaker, save current and start new
                            merged_segments.append({
                                "speaker": current_speaker,
                                "start": current_start,
                                "end": current_end
                            })
                            current_speaker = seg.speaker
                            current_start = seg.start
                            current_end = seg.end
                    
                    # Don't forget the last segment
                    if current_speaker is not None:
                        merged_segments.append({
                            "speaker": current_speaker,
                            "start": current_start,
                            "end": current_end
                        })
                    
                    # Now align words with speaker segments using sentence-aware logic
                    # First, build a flat list of all words with their timestamps
                    # Then split at sentence boundaries (punctuation) near speaker transitions
                    
                    # Simple approach: assign words to segments, but look for sentence endings
                    # to make cleaner speaker turns
                    diarization_result = []
                    
                    if merged_segments and word_timestamps:
                        # Create word assignments based on timing with sentence awareness
                        all_words = []
                        for word in word_timestamps:
                            all_words.append({
                                "word": word["word"],
                                "start": word["start"],
                                "end": word["end"]
                            })
                        
                        # Assign words to speakers, looking for sentence boundaries
                        current_seg_idx = 0
                        word_assignments = []
                        
                        for word_idx, word in enumerate(all_words):
                            word_start = word["start"]
                            
                            # Check if we should move to next speaker segment
                            while (current_seg_idx < len(merged_segments) - 1 and 
                                   word_start >= merged_segments[current_seg_idx + 1]["start"]):
                                current_seg_idx += 1
                            
                            word_assignments.append(current_seg_idx)
                        
                        # Now refine: if a word ends with punctuation and next word is different speaker
                        # keep it with current speaker (complete the sentence)
                        for i in range(len(word_assignments) - 1):
                            current_word = all_words[i]["word"].strip()
                            current_speaker = word_assignments[i]
                            next_speaker = word_assignments[i + 1]
                            
                            # If punctuation-ending word and speaker change, adjust
                            if current_speaker != next_speaker:
                                # If current word ends with sentence-ending punctuation,
                                # make sure to include it properly
                                if current_word.endswith(('?', '!', '.')):
                                    # Current word completes a sentence, assignment is correct
                                    pass
                                else:
                                    # Look ahead to find sentence ending in next few words
                                    # that should belong to current speaker
                                    for j in range(i + 1, min(i + 5, len(all_words))):
                                        check_word = all_words[j]["word"].strip()
                                        if check_word.endswith(('?', '!', '.')):
                                            # Include all words up to and including this one
                                            # in the current speaker's turn
                                            for k in range(i + 1, j + 1):
                                                word_assignments[k] = current_speaker
                                            break
                        
                        # Build speaker segments from word assignments
                        for seg_idx, seg in enumerate(merged_segments):
                            speaker_words = []
                            for word_idx, word in enumerate(all_words):
                                if word_assignments[word_idx] == seg_idx:
                                    speaker_words.append(word["word"])
                            
                            speaker_text = " ".join(speaker_words).strip()
                            # Clean up double spaces
                            while "  " in speaker_text:
                                speaker_text = speaker_text.replace("  ", " ")
                            
                            if speaker_text:
                                diarization_result.append({
                                    "speaker": f"Speaker {seg['speaker']}",
                                    "start": round(seg["start"], 2),
                                    "end": round(seg["end"], 2),
                                    "text": speaker_text
                                })
                    
                    logger.info(f"Diarization complete: {len(diarization_result)} speaker turns with text")
                else:
                    logger.info("No speaker segments detected")
                    
            except ImportError as e:
                logger.warning(f"Diarization unavailable - NeMo not installed: {e}")
            except Exception as e:
                logger.error(f"Diarization error: {e}", exc_info=True)
        
        response = {"transcript": full_transcript}
        if diarization_result:
            response["diarization"] = diarization_result
        
        return response
        
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
