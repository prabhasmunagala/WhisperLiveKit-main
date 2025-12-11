import io
import logging
import math
import sys
from typing import List

import numpy as np
import soundfile as sf

from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, lora_path=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_whisper_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)            
            if resolved_path.is_dir():
                model_info = detect_model_format(resolved_path)
                if not model_info.has_pytorch:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )            
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_whisper_model(str(resolved_path), lora_path=self.lora_path)

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_whisper_model(model_size, download_root=cache_dir, lora_path=self.lora_path)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        logger.debug(f"WhisperASR.transcribe: audio shape={audio.shape if hasattr(audio, 'shape') else 'N/A'}, "
                     f"audio type={type(audio)}, language={language}, options={options}")

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        
        logger.debug(f"WhisperASR.transcribe: result has {len(result.get('segments', []))} segments, "
                     f"text length={len(result.get('text', ''))}")
        if result.get('segments'):
            for i, seg in enumerate(result['segments'][:3]):  # Log first 3 segments
                logger.debug(f"  Segment {i}: text='{seg.get('text', '')[:50]}...', words={len(seg.get('words', []))}")
        
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(
                    word["start"],
                    word["end"],
                    word["word"],
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
                              

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                probability=word["probability"]
                token = ASRToken(word["start"], word["end"], word["word"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if not self.direct_english_translation and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True


class TransformersASR(ASRBase):
    """Uses Hugging Face Transformers pipeline as the backend."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is not available. Please install 'transformers' and 'torch'.")
            
        import torch
        import warnings

        model_id = model_dir if model_dir else model_size
        if not model_id:
            raise ValueError("Either model_size or model_dir must be set")

        logger.info(f"Loading Transformers pipeline with model {model_id}...")
        
        device = 0 if torch.cuda.is_available() else -1
        
        # Suppress the attention mask warning (it's harmless for Whisper)
        warnings.filterwarnings("ignore", message=".*attention_mask.*")
        
        # Load pipeline with automatic device placement
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        return self.pipe

    def transcribe(self, audio: np.ndarray, init_prompt: str = "", **kwargs) -> dict:
        # Prepare generation arguments
        # Default to word timestamps unless overridden
        if "return_timestamps" not in kwargs:
            kwargs["return_timestamps"] = "word"
            
        generate_kwargs = {}
        if self.original_language:
            generate_kwargs["language"] = self.original_language
            
        # Handle task
        task = self.transcribe_kargs.get("task", "transcribe")
        # Allow kwargs to override task if needed
        if "task" in kwargs:
            task = kwargs.pop("task")
            
        generate_kwargs["task"] = task
        
        # Warn about init_prompt if provided
        if init_prompt:
            logger.debug("TransformersASR: init_prompt is ignored in this implementation.")

        # Ensure audio is float32
        if hasattr(audio, "dtype") and audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Determine audio duration to optimize chunk size
        audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
        
        # For short audio (< 15s), don't use chunking for faster processing
        # For longer audio, use smaller chunks for real-time streaming
        if audio_duration < 15:
            # Short audio - process directly without chunking
            result = self.model(
                audio,
                generate_kwargs=generate_kwargs,
                **kwargs
            )
        else:
            # Longer audio - use smaller chunks for lower latency
            result = self.model(
                audio, 
                chunk_length_s=10,  # Reduced from 30 for faster streaming
                batch_size=1,  # Process one chunk at a time for lower latency
                generate_kwargs=generate_kwargs,
                **kwargs
            )
        return result

    def ts_words(self, result) -> List[ASRToken]:
        tokens = []
        # Transformers return_timestamps="word" returns chunks with word timestamps
        chunks = result.get("chunks", [])
        
        # For translation mode, timestamps might be None - estimate them
        audio_duration = 30.0  # Assume 30s chunks max
        num_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            timestamp = chunk.get("timestamp")
            
            # Handle timestamp extraction
            if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start, end = timestamp
                
                # If we have valid timestamps, use them
                if start is not None and end is not None:
                    tokens.append(ASRToken(start, end, text))
                elif start is not None:
                    # Only start time - estimate end
                    estimated_end = start + 0.5  # Assume ~0.5s per word
                    tokens.append(ASRToken(start, estimated_end, text))
                else:
                    # No valid timestamps - estimate based on position
                    if num_chunks > 0:
                        estimated_start = (i / num_chunks) * audio_duration
                        estimated_end = ((i + 1) / num_chunks) * audio_duration
                        tokens.append(ASRToken(estimated_start, estimated_end, text))
            else:
                # No timestamp at all - estimate based on position  
                if num_chunks > 0:
                    estimated_start = (i / num_chunks) * audio_duration
                    estimated_end = ((i + 1) / num_chunks) * audio_duration
                    tokens.append(ASRToken(estimated_start, estimated_end, text))
                    
        return tokens

    def segments_end_ts(self, result) -> List[float]:
        # Return the end timestamp of every chunk (word) since we don't have sentence segments
        chunks = result.get("chunks", [])
        end_times = []
        
        for i, c in enumerate(chunks):
            timestamp = c.get("timestamp")
            if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                end = timestamp[1]
                if end is not None:
                    end_times.append(end)
                else:
                    # Estimate end time
                    end_times.append((i + 1) * 0.5)  # ~0.5s per word estimate
            else:
                end_times.append((i + 1) * 0.5)
                
        return end_times

    def use_vad(self):
        pass
