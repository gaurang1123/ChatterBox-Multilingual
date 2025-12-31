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

# Create interface
with gr.Blocks(title="Multilingual TTS") as demo:
    gr.Markdown("# 🌍 Multilingual Text-to-Speech")
    gr.Markdown("Generate speech in 23 languages using Chatterbox Multilingual TTS")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Enter text in your chosen language...",
                lines=3
            )
            language_dropdown = gr.Dropdown(
                choices=[(f"{SUPPORTED_LANGUAGES[k]} ({k})", k) for k in SUPPORTED_LANGUAGES.keys()],
                label="Language",
                value="en"
            )
            generate_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            status_output = gr.Textbox(label="Status", interactive=False)
    
    # Connect the function
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, language_dropdown],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
