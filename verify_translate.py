import soundfile as sf
import numpy as np
import librosa

# Load Hindi audio
audio_path = "Hindi_M_Anuj.mp3"
print(f"Loading {audio_path}...")

audio, sr = sf.read(audio_path)
print(f"Loaded: {len(audio)} samples, {sr}Hz")

# Resample to 16kHz
if sr != 16000:
    print(f"Resampling from {sr}Hz to 16000Hz...")
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

# Convert to mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Convert to float32
audio = audio.astype(np.float32)

print(f"\nTesting Whisper translation...")
from whisperlivekit.whisper import load_model, transcribe

model = load_model("large-v3-turbo")
print("Model loaded")

# Test 1: Auto-detect language + translate
print("\n=== Test 1: language=None, task='translate' ===")
result1 = transcribe(model, audio, language=None, task="translate", verbose=False)
print(f"Result: {result1['text'][:200]}")
print(f"Detected language: {result1.get('language', 'N/A')}")

# Test 2: Explicit Hindi + translate
print("\n=== Test 2: language='hi', task='translate' ===")
result2 = transcribe(model, audio, language="hi", task="translate", verbose=False)
print(f"Result: {result2['text'][:200]}")
print(f"Language: {result2.get('language', 'N/A')}")

# Test 3: Just transcribe (for comparison)
print("\n=== Test 3: language='hi', task='transcribe' ===")
result3 = transcribe(model, audio, language="hi", task="transcribe", verbose=False)
print(f"Result: {result3['text'][:200]}")
print(f"Language: {result3.get('language', 'N/A')}")

print("\n" + "="*60)
print("If Test 1 and Test 2 show Hindi text, then Whisper's")
print("translate task is not working for this model/audio combination.")
print("="*60)
