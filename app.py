
import subprocess
subprocess.run('pip install flash-attn==2.7.4.post1 pandas --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
subprocess.run('pip install git+https://github.com/RoyalCities/RC-stable-audio-tools.git --no-deps', shell=True)

import spaces
import os
import json
import math
import random
import tempfile
import numpy as np
import torch
import torchaudio
import gradio as gr
from einops import rearrange
from huggingface_hub import hf_hub_download

# Stable Audio Tools imports
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.interface.prompts.master_prompt_map import prompt_generator_foundation

# ==========================================
# 1. GLOBAL INITIALIZATION & MODEL PRELOADING
# ==========================================
MODEL_REPO = "RoyalCities/Foundation-1"
MODEL_FILE = "Foundation_1.safetensors" # Explicitly match the file in the repo

# Automatically choose CUDA if available, fallback to MPS or CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

print(f"Loading {MODEL_REPO} onto {DEVICE} ({DTYPE})...")

def load_custom_model(repo_id, filename):
    """Custom loader to handle non-standard model filenames."""
    # 1. Download config
    config_path = hf_hub_download(repo_id, filename="model_config.json", repo_type='model')
    with open(config_path) as f:
        model_config = json.load(f)
        
    # 2. Create model from config
    model = create_model_from_config(model_config)
    
    # 3. Download and load weights
    ckpt_path = hf_hub_download(repo_id, filename=filename, repo_type='model')
    model.load_state_dict(load_ckpt_state_dict(ckpt_path))
    
    return model, model_config

# Preload model globally
model, model_config = load_custom_model(MODEL_REPO, MODEL_FILE)
model = model.to(DEVICE).to(DTYPE).eval().requires_grad_(False)

SAMPLE_RATE = model_config.get("sample_rate", 44100)
MIN_INPUT_LENGTH = getattr(model, "min_input_length", None)

print("Model successfully loaded and ready for inference!")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_target_samples(bars: int, bpm: float):
    """Calculates the exact sample length and padded model input size."""
    # Calculate exact clip length in seconds and samples
    clip_seconds = (60.0 / float(bpm)) * 4.0 * float(bars) # 4 beats per bar
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))
    
    # Calculate padded length for the model (must be aligned to min_input_length)
    seconds_total_int = int(math.ceil(clip_samples / SAMPLE_RATE))
    target_samples = int(seconds_total_int * SAMPLE_RATE)
    
    if isinstance(MIN_INPUT_LENGTH, int) and MIN_INPUT_LENGTH > 0 and (target_samples % MIN_INPUT_LENGTH) != 0:
        target_samples += (MIN_INPUT_LENGTH - (target_samples % MIN_INPUT_LENGTH))
        
    return clip_samples, seconds_total_int, target_samples

def generate_random_prompt(is_experimental: bool):
    """Uses the built-in prompt generator to create a Foundation-style prompt."""
    variant = "T1" if is_experimental else "M1"
    prompt = prompt_generator_foundation(
        variant=variant, 
        mode="standard", 
        allow_timbre_mix=is_experimental
    )
    
    bars = random.choice([4, 8])
    bpm = random.choice([100, 110, 120, 128, 130, 140, 150])
    note = random.choice(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
    scale = random.choice(["major", "minor"])
    
    return prompt, bars, bpm, note, scale

# ==========================================
# 3. CORE GENERATION LOGIC
# ==========================================
@spaces.GPU
def generate_audio(prompt, negative_prompt, bars, bpm, note, scale, steps, cfg_scale, seed):
    try:
        seed_val = int(seed) if seed != -1 else random.randint(0, 2**32 - 1)
        amended_prompt = f"{prompt}, {note} {scale}, {bars} bars, {bpm}BPM,"
        
        clip_samples, seconds_total_int, input_sample_size = calculate_target_samples(bars, bpm)
        
        conditioning = [{"prompt": amended_prompt, "seconds_start": 0.0, "seconds_total": float(seconds_total_int)}]
        neg_conditioning = [{"prompt": negative_prompt, "seconds_start": 0.0, "seconds_total": float(seconds_total_int)}] if negative_prompt else None
        
        audio = generate_diffusion_cond(
            model,
            conditioning=conditioning,
            negative_conditioning=neg_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=1,
            sample_size=input_sample_size,
            sample_rate=SAMPLE_RATE,
            seed=seed_val,
            device=DEVICE,
            sampler_type="dpmpp-3m-sde",
            sigma_min=0.03,
            sigma_max=1000,
        )
        
        # 1. Rearrange to [channels, samples] and ensure float32
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)

        # 2. PEAK NORMALIZATION — prevents clipping distortion
        # The diffusion model outputs can far exceed [-1, 1].
        # Hard-clamping (the repo default) chops those peaks → audible distortion.
        # Instead, scale the whole waveform so the loudest peak sits at -1 dBFS
        # (≈0.89), leaving headroom to avoid inter-sample clipping in DAWs.
        max_amp = torch.abs(audio).max()
        if max_amp > 1e-8:
            audio = audio / max_amp * 0.89125  # -1 dBFS headroom

        # 3. Trim to the exact deterministic grid length
        end = min(int(audio.shape[-1]), int(clip_samples))
        audio = audio[:, :max(1, end)].contiguous()

        # 4. Apply a tiny 15ms fade-out to avoid clicks at the end
        fade_ms = 15.0
        fade_len = int(round((fade_ms / 1000.0) * SAMPLE_RATE))
        if fade_len > 1 and audio.shape[-1] > 1:
            fade_len = min(fade_len, audio.shape[-1])
            ramp = torch.linspace(1.0, 0.0, steps=fade_len, device=audio.device)
            audio[:, -fade_len:] *= ramp

        # 5. Save as float32 WAV — let torchaudio handle bit-depth encoding
        audio = audio.clamp(-1, 1).cpu()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        torchaudio.save(output_path, audio, SAMPLE_RATE)
        return output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An error occurred during generation: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==========================================
# 4. GRADIO UI
# ==========================================
CSS = """
footer { display: none !important; }

/* ── Hero header ── */
#hero {
    text-align: center;
    padding: 2.8rem 1rem 1.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 0.5rem;
}
#hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem;
    background: linear-gradient(90deg, #a78bfa 0%, #818cf8 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Generate button ── */
#gen-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    border: none !important;
    height: 56px !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
#gen-btn:hover {
    box-shadow: 0 0 28px rgba(124, 58, 237, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Audio output panel ── */
#audio-panel {
    border-radius: 14px !important;
    overflow: hidden;
}
"""

with gr.Blocks(
    title="local-offline-ai-music-generator",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css=CSS,
) as demo:

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hero">
      <h1>local-offline-ai-music-generator</h1>
      <p>Diffusion-based music generation &nbsp;·&nbsp; powered by <a href="https://huggingface.co/RoyalCities/Foundation-1" target="_blank" style="color:#a78bfa;text-decoration:none;">Foundation-1</a></p>
    </div>
    """)

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left: controls ────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=420):

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. deep house piano loop, warm chords, late night atmosphere, lush reverb ...",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Optional: elements to avoid",
                value="",
            )

            # Musical parameters — all 4 in one row
            gr.Markdown("**Musical Parameters**")
            with gr.Row():
                note  = gr.Dropdown(choices=["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"], value="C",   label="Key")
                scale = gr.Dropdown(choices=["major","minor"], value="minor", label="Scale")
                bars  = gr.Dropdown(choices=[4, 8], value=8, label="Bars")
                bpm   = gr.Dropdown(choices=[100,110,120,128,130,140,150], value=128, label="BPM")

            with gr.Accordion("Advanced Settings", open=False):
                steps     = gr.Slider(minimum=10, maximum=150, value=75, step=1,    label="Inference Steps", info="Higher = better quality, slower")
                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.1, label="CFG Scale", info="How closely the model follows your prompt")
                seed      = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

            with gr.Row(equal_height=True):
                experimental_mode = gr.Checkbox(
                    label="Experimental Prompts (More Timbre Mix)",
                    value=False,
                    scale=2,
                )
                random_btn = gr.Button("🎲  Random Prompt", variant="secondary", scale=3)

            generate_btn = gr.Button("🚀  Generate Audio", variant="primary", size="lg", elem_id="gen-btn")

        # ── Right: output ─────────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=300):
            audio_out = gr.Audio(
                label="Generated Audio",
                type="filepath",
                elem_id="audio-panel",
            )
            gr.Markdown("""
**Prompt tips**
- Name specific instruments: *Rhodes piano, 808 bass, Juno pad*
- Add mood/texture: *warm, dusty, melancholic, driving*
- Avoid generic words like *good* or *nice*

**Parameter guide**
- **4 bars** ≈ 8 s &nbsp;|&nbsp; **8 bars** ≈ 16 s
- **Steps 50–75** balances speed & quality
- **CFG 5–9** gives the most musical results
""")

    # ── Event wiring ──────────────────────────────────────────────────────────
    random_btn.click(
        fn=generate_random_prompt,
        inputs=[experimental_mode],
        outputs=[prompt, bars, bpm, note, scale],
    )
    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt, negative_prompt, bars, bpm, note, scale, steps, cfg_scale, seed],
        outputs=[audio_out],
    )

# Start the app
if __name__ == "__main__":
    demo.queue().launch()