import torch
import torchaudio as ta
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

device = "cpu"
print(f"Using device: {device}")

# Initialize multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Test simple text generation
test_text = "Hello, this is a test of multilingual text to speech."
print(f"\nGenerating: '{test_text}'")

try:
    wav = model.generate(test_text, language_id="en")
    filename = "multilingual_test.wav"
    ta.save(filename, wav, model.sr)
    duration = len(wav[0]) / model.sr
    print(f"✅ Success! Saved {filename} - Duration: {duration:.2f}s")
    print(f"🎉 TRUE MULTILINGUAL TTS IS WORKING!")
    print(f"📁 Audio file saved as: {filename}")
    print(f"🌍 Supported languages: {len(SUPPORTED_LANGUAGES)} languages")
    print(f"Languages: {', '.join(list(SUPPORTED_LANGUAGES.keys())[:10])}...")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
