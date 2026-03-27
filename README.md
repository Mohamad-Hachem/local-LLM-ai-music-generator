# local-offline-ai-music-generator by Foundation-1

> Diffusion-based AI music generation with precise musical control — key, scale, BPM, and bar-length — powered by the Foundation-1 model and deployed on Hugging Face Spaces.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.10.0-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![Model](https://img.shields.io/badge/Model-Foundation--1-a78bfa)](https://huggingface.co/RoyalCities/Foundation-1)

---

## What Is This?

Foundation-1 Music Generator is a Gradio web application that wraps the **Foundation-1 diffusion model** — a latent audio diffusion model built on the `stable-audio-tools` framework — into a clean, musician-friendly interface.

Most text-to-audio tools treat musical structure as an afterthought, burying tempo and key inside the prompt text and hoping the model infers them. This app takes a different approach: BPM, key, scale, and bar count are **first-class parameters** that get baked directly into the conditioning signal, so the model generates clips that are metrically and harmonically grounded from the start.

The output is a peak-normalized, bar-accurate WAV file at 44.1 kHz — ready to drop into a DAW without manual trimming or gain staging.

| File | Responsibility |
|---|---|
| `app.py` | Entire application — model loading, generation logic, audio post-processing, Gradio UI |
| `requirements.txt` | Pinned dependency set for reproducible HF Spaces deployment |
| `.gitattributes` | Git LFS tracking for large binary assets (`.safetensors`, `.pt`, `.ckpt`) |

---

## Screenshots

### Default UI — empty state

![Default UI](application-screenshots/Screenshot%202026-03-27%20150630.png)

The interface on load: prompt and negative-prompt fields, musical parameter dropdowns (Key, Scale, Bars, BPM), a collapsible **Advanced Settings** accordion, the **Experimental Prompts** toggle, and the **Random Prompt** / **Generate Audio** buttons. The right panel awaits output.

---

### Random prompt populated (Experimental mode enabled)

![Random prompt filled](application-screenshots/Screenshot%202026-03-27%20150643.png)

Clicking **🎲 Random Prompt** with **Experimental Prompts** enabled draws from the T1 variant of the built-in `prompt_generator_foundation`. The prompt, bars, BPM, key, and scale are all populated in one click — useful for exploring the model's range without writing prompts manually.

---

### Generated audio with waveform

![Generated audio output](application-screenshots/Screenshot%202026-03-27%20150708.png)

After generation completes, the **Generated Audio** panel renders an interactive waveform with playback controls and a download button. The clip is bar-accurate (trimmed to the exact `bars × BPM` grid) and peak-normalized to −1 dBFS.

---

## Feature List

### Music Generation
- Text-conditioned audio generation using the Foundation-1 diffusion model
- **Key selection** — all 12 chromatic notes (A through G#)
- **Scale selection** — major or minor
- **Bar count** — 4 bars (~8 s) or 8 bars (~16 s)
- **BPM** — discrete choices from 100 to 150 (100, 110, 120, 128, 130, 140, 150)
- **Negative prompt** support to steer the diffusion away from unwanted elements
- **Seed control** — reproducible generations or random on `-1`

### Diffusion Controls
- **Inference steps** — 10 to 150; ~75 steps balances speed and quality
- **CFG scale** — 1.0 to 15.0; 5–9 gives the most musical results
- **Sampler** — DPM++ 3M SDE with `sigma_min=0.03`, `sigma_max=1000`

### Prompt Tooling
- **Random Prompt** button — draws from the built-in `prompt_generator_foundation` utility to generate Foundation-style descriptive prompts
- **Experimental mode (T1)** — enables the T1 prompt variant, which allows richer timbre mixing compared to the standard M1 variant

### Audio Post-Processing
- **Peak normalization to −1 dBFS** (scale factor ≈ 0.89125) — prevents inter-sample clipping without the audible distortion caused by hard-clamping
- **Bar-accurate trimming** — output is cut to the exact sample count implied by `bars × BPM × 44100 Hz`, eliminating diffusion tail silence
- **15 ms linear fade-out** — removes end-of-clip clicks
- Output saved as 32-bit float WAV via `torchaudio`

### Deployment & Hardware
- `@spaces.GPU` decorator for zero-config GPU allocation on Hugging Face Spaces
- Automatic device fallback: CUDA (bfloat16/float16) → Apple MPS (float16) → CPU (float32)
- Model loaded once at startup and held in memory; `torch.cuda.empty_cache()` called after each generation

---

## How It Works

1. **Startup** — `app.py` pip-installs `flash-attn` (with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`) and the custom `RC-stable-audio-tools` fork at import time, before any model code runs.

2. **Model loading** — `load_custom_model()` fetches `model_config.json` from the `RoyalCities/Foundation-1` HuggingFace Hub repo, instantiates the model architecture via `create_model_from_config`, then downloads and loads `Foundation_1.safetensors` weights. The model is moved to the selected device and dtype, then set to `eval()` with gradients disabled.

3. **User input** — The Gradio UI collects a text prompt, an optional negative prompt, and musical/diffusion parameters. Clicking **Generate Audio** calls `generate_audio()`.

4. **Prompt amendment** — The raw prompt is extended: `"{user prompt}, {note} {scale}, {bars} bars, {bpm}BPM,"`. This structured suffix is fed directly into the conditioning dict alongside `seconds_start` and `seconds_total` timestamps.

5. **Sample size calculation** — `calculate_target_samples()` converts bars + BPM to an exact clip length in samples, then pads to the model's `min_input_length` alignment requirement.

6. **Diffusion** — `generate_diffusion_cond()` runs the DPM++ 3M SDE sampler for the configured number of steps, producing a raw waveform tensor.

7. **Post-processing** — The waveform is rearranged to `[channels, samples]`, peak-normalized to −1 dBFS, trimmed to the exact musical grid, and faded out over the last 15 ms.

8. **Output** — The processed tensor is saved to a `tempfile` WAV and returned to the Gradio `Audio` component for playback and download.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Gradio UI (app.py)                  │
│                                                          │
│  [Prompt]  [Neg. Prompt]  [Key] [Scale] [Bars] [BPM]   │
│  [Steps]   [CFG Scale]    [Seed]  [Experimental Mode]   │
│                    │                                     │
│            [Generate Audio]                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   generate_audio()    │  ← @spaces.GPU
         │                       │
         │  1. Amend prompt      │
         │  2. Calc sample size  │
         │  3. Build conditioning│
         └──────────┬────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  generate_diffusion_cond()   │
     │  (stable-audio-tools)        │
     │                              │
     │  Sampler: DPM++ 3M SDE       │
     │  Model:   Foundation-1       │
     │  Weights: .safetensors       │
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │     Audio Post-Processing    │
     │                              │
     │  Peak normalize → −1 dBFS    │
     │  Trim to bars×BPM grid       │
     │  15 ms fade-out              │
     │  Clamp → float32 WAV         │
     └──────────────┬───────────────┘
                    │
                    ▼
          [Audio Output Widget]
         (playback + download)
```

**Model weight acquisition** (first run only):

```
HuggingFace Hub
  └── RoyalCities/Foundation-1
        ├── model_config.json   → architecture definition
        └── Foundation_1.safetensors → weights (~cached locally)
```

---

## Project Structure

```
Foundation-1/
├── app.py               # Entire application (model loader, generation, Gradio UI)
├── requirements.txt     # Pinned dependencies for HF Spaces deployment
├── .gitattributes       # Git LFS rules for large binary files
└── README.md            # This file / HF Spaces YAML frontmatter
```

---

## Installation

### Prerequisites

- Python 3.12
- CUDA-capable GPU recommended (NVIDIA with CUDA 12.4); CPU inference is supported but slow
- Git LFS installed (`git lfs install`) if cloning model weights locally

### Setup

```bash
# Clone the repository
git clone https://github.com/Mohamad-Hachem/Foundation-1.git
cd Foundation-1

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `flash-attn` and the custom `stable-audio-tools` fork are installed automatically when `app.py` first runs via `subprocess.run(...)` at the top of the file. You do not need to install them manually.

> **Note:** On first launch, `Foundation_1.safetensors` (~several GB) is downloaded from `RoyalCities/Foundation-1` on the HuggingFace Hub and cached in your local HF cache directory (`~/.cache/huggingface/hub`). Subsequent runs use the cached file.

---

## Usage

```bash
# Start the Gradio app (auto-reloads on file changes during development)
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it in your browser.

### Basic workflow

1. Type a descriptive prompt in the **Prompt** field (see tips in the UI).
2. Set **Key**, **Scale**, **Bars**, and **BPM** to match your intended musical context.
3. Optionally fill in a **Negative Prompt** to suppress unwanted sounds.
4. Click **Generate Audio**.
5. The generated WAV is playable directly in the browser and downloadable.

### Using the random prompt generator

Click **🎲 Random Prompt** to populate all fields with a Foundation-style prompt and randomized musical parameters. Enable **Experimental Prompts** first to use the T1 variant, which allows richer timbre blending.

---

## Example Use Cases

| Goal | Prompt example | Params |
|---|---|---|
| Lofi hip-hop loop | `dusty Rhodes piano, vinyl crackle, mellow chords, lofi hip-hop` | C minor, 4 bars, 90 BPM |
| Tech house groove | `punchy 808 kick, syncopated claps, driving acid bass, tech house` | A minor, 8 bars, 128 BPM |
| Atmospheric pad | `lush analog string pad, slow attack, reverb tail, ambient` | F# major, 8 bars, 110 BPM |
| Melodic trap | `melodic piano, glassy 808, trap hi-hats, melancholic` | D minor, 4 bars, 140 BPM |

**Advanced — CFG and steps tuning:**

```
Steps 30–50  → fast drafts, exploratory generation
Steps 75     → recommended default; good quality/speed balance
Steps 100+   → diminishing returns; use for final renders

CFG 4–6      → more variation, looser prompt adherence
CFG 7–9      → recommended; tracks the prompt closely
CFG 10+      → can over-saturate; use with specific prompts
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI framework** | Gradio 6.10.0 |
| **Diffusion engine** | `stable-audio-tools` (RoyalCities custom fork) |
| **Model weights** | Foundation-1 (`Foundation_1.safetensors`) via HuggingFace Hub |
| **Sampler** | DPM++ 3M SDE |
| **Deep learning** | PyTorch 2.6.0+cu124 |
| **Audio I/O** | torchaudio 2.6.0+cu124 |
| **Tensor utilities** | einops, numpy |
| **Model serialization** | safetensors |
| **GPU allocation** | `spaces` (HuggingFace Spaces decorator) |
| **Runtime** | Python 3.12 |
| **Deployment** | HuggingFace Spaces (Gradio SDK) |

---

## Limitations

- **Bar/BPM choices are discrete** — BPM is limited to 7 preset values (100–150) and bars to 4 or 8. Arbitrary tempo and length are not currently supported.
- **Single-clip generation** — batch size is fixed at 1; no multi-generation or variation comparison in a single run.
- **GPU memory** — inference at full quality (150 steps, 8 bars) is memory-intensive. On consumer GPUs with <8 GB VRAM, lower steps or use 4 bars.
- **Cold start** — first run downloads model weights from the HuggingFace Hub, which can take several minutes depending on network speed.
- **CPU inference is slow** — without a CUDA or MPS device, generation at 75 steps takes significantly longer than real-time.
- **No streaming** — the full clip must complete before audio is returned to the UI; there is no progressive output.
- **Model scope** — Foundation-1 is oriented toward loop and stem generation. Full-arrangement or long-form composition is outside its design intent.

---

## Future Improvements

- **Arbitrary BPM and bar count** — replace discrete dropdowns with free-entry numeric inputs and update the sample-size math accordingly.
- **Variation count** — add a slider to generate N variations in a single click and present them as a comparison gallery.
- **Stereo/mono toggle** — expose channel configuration to the user.
- **In-browser waveform visualizer** — render a waveform or spectrogram alongside the audio player.
- **Prompt history** — persist recent prompts and their outputs in-session for quick recall and iteration.
- **Export metadata** — embed prompt, BPM, key, and seed into WAV `INFO` tags for DAW import traceability.
- **Progress indicator** — stream step-by-step diffusion progress back to the UI rather than blocking until completion.
- **Quantized inference** — explore INT8/FP8 quantization to reduce VRAM footprint and improve CPU throughput.
