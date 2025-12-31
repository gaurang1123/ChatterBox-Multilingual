import gradio as gr
import torch
import torchaudio as ta
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import tempfile
import os
import re

# Initialize model
device = "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

def generate_speech(text, language, reference_audio, exaggeration, cfg_weight, speed_factor, temperature):
    """Generate speech with advanced parameters"""
    if not text.strip():
        return None, "Please enter some text"
    
    try:
        # Prepare parameters
        kwargs = {
            "language_id": language,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature
        }
        
        # Add reference audio if provided
        if reference_audio is not None:
            kwargs["audio_prompt_path"] = reference_audio
        
        # Generate audio
        wav = model.generate(text, **kwargs)
        
        # Apply speed control if needed
        if speed_factor != 1.0:
            import librosa
            wav_np = wav[0].cpu().numpy()
            wav_stretched = librosa.effects.time_stretch(wav_np, rate=speed_factor)
            wav = torch.tensor(wav_stretched).unsqueeze(0)
        
        # Save to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        ta.save(output_path, wav, model.sr)
        
        duration = len(wav[0]) / model.sr
        lang_name = SUPPORTED_LANGUAGES[language]
        status = f"✅ Generated {duration:.1f}s of {lang_name} speech"
        if reference_audio:
            status += " (with voice cloning)"
        
        return output_path, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def chunk_text(text, max_chars=500):
    """Split text into chunks at sentence boundaries"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text_file(file, language, reference_audio, exaggeration, cfg_weight, speed_factor, temperature, chunk_size):
    """Process uploaded text file in chunks"""
    if file is None:
        return None, "Please upload a text file"
    
    try:
        # Read file content
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) < 100:
            return None, "Text file too short (minimum 100 characters)"
        
        # Split into chunks
        chunks = chunk_text(text, chunk_size)
        
        if not chunks:
            return None, "No valid text chunks found"
        
        # Generate audio for each chunk
        audio_segments = []
        total_duration = 0
        
        for i, chunk in enumerate(chunks):
            kwargs = {
                "language_id": language,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature
            }
            
            if reference_audio is not None:
                kwargs["audio_prompt_path"] = reference_audio
            
            # Generate audio for chunk
            wav = model.generate(chunk, **kwargs)
            
            # Apply speed control
            if speed_factor != 1.0:
                import librosa
                wav_np = wav[0].cpu().numpy()
                wav_stretched = librosa.effects.time_stretch(wav_np, rate=speed_factor)
                wav = torch.tensor(wav_stretched).unsqueeze(0)
            
            audio_segments.append(wav[0])
            total_duration += len(wav[0]) / model.sr
        
        # Concatenate all audio segments
        final_audio = torch.cat(audio_segments, dim=0).unsqueeze(0)
        
        # Save final audio
        output_path = tempfile.mktemp(suffix=".wav")
        ta.save(output_path, final_audio, model.sr)
        
        lang_name = SUPPORTED_LANGUAGES[language]
        status = f"✅ Processed {len(chunks)} chunks, {total_duration:.1f}s of {lang_name} speech"
        if reference_audio:
            status += " (with voice cloning)"
        
        return output_path, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Language choices
language_choices = [(f"{SUPPORTED_LANGUAGES[k]} ({k})", k) for k in SUPPORTED_LANGUAGES.keys()]

# Create advanced interface
with gr.Blocks(title="Chatterbox Multilingual TTS") as demo:
    gr.Markdown("# 🌍 Chatterbox Multilingual TTS")
    gr.Markdown("Advanced text-to-speech with voice cloning and parameter control")
    
    with gr.Tabs():
        with gr.TabItem("Text Input"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to speak",
                        placeholder="Enter text in your chosen language...",
                        lines=4
                    )
                    
                    language_dropdown = gr.Dropdown(
                        choices=language_choices,
                        label="Language",
                        value="en"
                    )
                    
                    reference_audio = gr.Audio(
                        label="Reference Voice (optional)",
                        type="filepath",
                        help="Upload an audio file to clone the voice"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Parameters")
                    
                    exaggeration = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Exaggeration",
                        info="Emotion intensity (0.0 = neutral, 1.0 = very expressive)"
                    )
                    
                    cfg_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="CFG Weight",
                        info="Guidance strength (lower = more natural pacing)"
                    )
                    
                    speed_factor = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed",
                        info="Speech speed (1.0 = normal)"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Randomness (0.1 = deterministic, 1.0 = creative)"
                    )
                    
                    generate_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")
            
            with gr.Row():
                audio_output = gr.Audio(label="Generated Speech")
                status_output = gr.Textbox(label="Status", interactive=False)
            
            # Examples
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    ["Hello, this is a test of English speech.", "en", None, 0.5, 0.5, 1.0, 0.7],
                    ["Hola, esto es una prueba de voz en español.", "es", None, 0.7, 0.3, 1.0, 0.7],
                    ["Bonjour, ceci est un test de synthèse vocale française.", "fr", None, 0.5, 0.5, 0.9, 0.7],
                    ["नमस्ते, यह हिंदी में बोलने का परीक्षण है।", "hi", None, 0.6, 0.4, 1.0, 0.7],
                ],
                inputs=[text_input, language_dropdown, reference_audio, exaggeration, cfg_weight, speed_factor, temperature]
            )
            
            # Connect the function
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, language_dropdown, reference_audio, exaggeration, cfg_weight, speed_factor, temperature],
                outputs=[audio_output, status_output]
            )
        
        with gr.TabItem("File Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Text File",
                        file_types=[".txt"],
                        help="Upload a .txt file (supports large files 40K+ characters)"
                    )
                    
                    file_language_dropdown = gr.Dropdown(
                        choices=language_choices,
                        label="Language",
                        value="en"
                    )
                    
                    file_reference_audio = gr.Audio(
                        label="Reference Voice (optional)",
                        type="filepath",
                        help="Upload an audio file to clone the voice"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Parameters")
                    
                    chunk_size = gr.Slider(
                        minimum=200,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Chunk Size",
                        info="Characters per chunk (smaller = more natural pauses)"
                    )
                    
                    file_exaggeration = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Exaggeration",
                        info="Emotion intensity"
                    )
                    
                    file_cfg_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="CFG Weight",
                        info="Guidance strength"
                    )
                    
                    file_speed_factor = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed",
                        info="Speech speed"
                    )
                    
                    file_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Randomness"
                    )
                    
                    process_btn = gr.Button("📄 Process File", variant="primary", size="lg")
            
            with gr.Row():
                file_audio_output = gr.Audio(label="Generated Audio")
                file_status_output = gr.Textbox(label="Status", interactive=False)
            
            # Connect file processing function
            process_btn.click(
                fn=process_text_file,
                inputs=[file_input, file_language_dropdown, file_reference_audio, file_exaggeration, file_cfg_weight, file_speed_factor, file_temperature, chunk_size],
                outputs=[file_audio_output, file_status_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
