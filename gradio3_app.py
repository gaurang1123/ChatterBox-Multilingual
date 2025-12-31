import gradio as gr
import torch
import torchaudio as ta
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Initialize model
device = "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

def generate_speech(text, language):
    """Generate speech from text in the specified language"""
    if not text.strip():
        return None, "Please enter some text"
    
    try:
        # Generate audio
        wav = model.generate(text, language_id=language)
        
        # Save to temporary file
        output_path = "output.wav"
        ta.save(output_path, wav, model.sr)
        
        return output_path, f"✅ Generated {len(wav[0])/model.sr:.1f}s of {SUPPORTED_LANGUAGES[language]} speech"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create language choices for dropdown
language_choices = [(f"{SUPPORTED_LANGUAGES[k]} ({k})", k) for k in SUPPORTED_LANGUAGES.keys()]

# Create interface using Gradio 3.x syntax
demo = gr.Interface(
    fn=generate_speech,
    inputs=[
        gr.Textbox(
            label="Text to speak",
            placeholder="Enter text in your chosen language...",
            lines=3
        ),
        gr.Dropdown(
            choices=language_choices,
            label="Language",
            value="en"
        )
    ],
    outputs=[
        gr.Audio(label="Generated Speech"),
        gr.Textbox(label="Status")
    ],
    title="🌍 Multilingual Text-to-Speech",
    description="Generate speech in 23 languages using Chatterbox Multilingual TTS"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
