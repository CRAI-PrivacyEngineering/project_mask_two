# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install PyTorch (choose based on your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# OR
pip install torch torchvision  # CPU only

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration (2 minutes)

Edit `config.yaml`:

```yaml
sdxl:
  lora_path: "/path/to/your/lora.safetensors"  # ← Optional: Update if using LoRA
  # Or use HuggingFace LoRA: "username/lora-name"

i2v:
  method: "svd"  # ← Uses Stable Video Diffusion (recommended, auto-downloads)
  # Or use "wan" for Wan2.2 (requires custom setup)
```

**Note**: SVD models download automatically. LoRA is optional.

## Usage

### Command Line

```bash
python main.py input.jpg "a beautiful landscape, cinematic, high quality"
```

### Python Script

```python
from src.pipeline import VideoGenerationPipeline

pipeline = VideoGenerationPipeline()
video_path = pipeline.generate(
    input_image="input.jpg",
    prompt="your prompt here"
)
```

## Pipeline Steps

1. **SDXL img2img** → Transforms your reference image based on prompt
2. **Image Upscale** → 4x upscaling for quality (Real-ESRGAN)
3. **I2V Conversion** → Converts to video frames (SVD or Wan2.2)
4. **Video Upscale** → Final quality enhancement

## Output Location

All outputs saved to `./outputs/` directory:
- `*_step1_img2img.png` - After SDXL generation
- `*_step2_upscaled.png` - After image upscaling
- `*_step3_frames/` - Video frames from Wan I2V
- `*_final.mp4` - Final video

## Tips

- **Consistency**: Set `seed` in config for reproducible results
- **Quality**: Increase `num_inference_steps` (slower but better)
- **Memory**: Use `float16` and reduce image size if OOM
- **Motion**: Adjust `motion_bucket_id` in Wan I2V config

## Troubleshooting

**Out of memory?**
- Reduce `num_inference_steps`
- Use smaller input images
- Enable VAE slicing (already enabled)

**Model not found?**
- SVD downloads automatically - check internet connection
- LoRA: Check path in `config.yaml` or use HuggingFace ID
- Wan2.2: Requires custom setup (see `src/i2v.py`)

**Slow generation?**
- Use GPU (`device: "cuda"`)
- Install `xformers` for faster attention
- Reduce `num_frames` in Wan I2V config

## Next Steps

- Read `README.md` for detailed documentation
- Check `SETUP.md` for advanced setup
- See `example.py` for code examples

