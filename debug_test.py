import sys
import numpy as np
import soundfile as sf

# Load audio file
audio_path = "LIVEKIT-+914045307490-20251202-194240-+917075176499.wav"
print(f"Loading {audio_path}...")

try:
    audio, sr = sf.read(audio_path)
    print(f"✓ Audio loaded: {len(audio)} samples, {sr}Hz, duration={len(audio)/sr:.2f}s")
    print(f"  Shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"  Min: {audio.min():.4f}, Max: {audio.max():.4f}, Mean: {audio.mean():.4f}")
    
    # Check if audio is silent
    if np.abs(audio).max() < 0.001:
        print("⚠ WARNING: Audio appears to be silent or very quiet!")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"  Resampling from {sr}Hz to 16000Hz...")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
        print(f"  ✓ Resampled: {len(audio)} samples")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print(f"  Converting from {audio.shape[1]} channels to mono...")
        audio = audio.mean(axis=1)
        print(f"  ✓ Converted to mono")
    
    # Convert to float32 (required by Whisper)
    if audio.dtype != np.float32:
        print(f"  Converting from {audio.dtype} to float32...")
        audio = audio.astype(np.float32)
        print(f"  ✓ Converted to float32")
    
    # Test transcription
    print("\nTesting transcription...")
    from whisperlivekit.whisper import load_model, transcribe
    
    print("Loading model...")
    model = load_model("large-v3-turbo")
    print("✓ Model loaded")
    
    print("\nTranscribing (translate task)...")
    result = transcribe(model, audio, language="te", task="translate", word_timestamps=True)
    
    print(f"\n✓ Transcription complete!")
    print(f"  Language detected: {result.get('language', 'N/A')}")
    print(f"  Number of segments: {len(result.get('segments', []))}")
    print(f"  Text length: {len(result.get('text', ''))}")
    print(f"\nFull text:\n{result.get('text', '(empty)')}")
    
    if result.get('segments'):
        print(f"\nFirst 3 segments:")
        for i, seg in enumerate(result['segments'][:3]):
            print(f"  [{i}] {seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text']}")
            print(f"      Words: {len(seg.get('words', []))}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
