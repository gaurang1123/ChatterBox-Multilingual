import torch
import torchaudio as ta
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

device = "cpu"
print(f"Using device: {device}")
print(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")

# Initialize multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Test different languages with simple phrases
test_cases = [
    ("en", "Hello, this is English."),
    ("es", "Hola, esto es español."),
    ("hi", "नमस्ते, यह हिंदी है।"),
    ("zh", "你好，这是中文。"),
]

print("\nTesting TRUE multilingual TTS...")
for lang_code, text in test_cases:
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
    print(f"\nGenerating {lang_name} ({lang_code}): '{text}'")
    
    try:
        wav = model.generate(text, language=lang_code)
        filename = f"true_multilingual_{lang_code}.wav"
        ta.save(filename, wav, model.sr)
        duration = len(wav[0]) / model.sr
        print(f"✅ Saved {filename} - Duration: {duration:.2f}s")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "="*60)
print("🎉 TRUE MULTILINGUAL TTS TEST COMPLETE!")
print("Listen to verify actual language pronunciation!")
print("="*60)
