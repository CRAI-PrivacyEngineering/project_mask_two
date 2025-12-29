# Recent Updates and Configuration Guide

## What's New

### ✅ Complete Pipeline Implementation
- **SDXL LoRA img2img**: Full support for loading LoRA weights from files or HuggingFace
- **Image Upscaling**: Real-ESRGAN integration with automatic model downloads
- **Image-to-Video**: Support for both Stable Video Diffusion (SVD) and Wan2.2
- **Video Upscaling**: Frame-by-frame upscaling with interpolation

### ✅ Model Integration
- **Stable Video Diffusion (SVD)**: Fully integrated, auto-downloads from HuggingFace
- **SDXL**: Auto-downloads base model, supports LoRA from files or HuggingFace
- **Real-ESRGAN**: Auto-downloads models on first use
- **Wan2.2**: Placeholder structure ready for integration

### ✅ Reference Image Support
- Accepts image paths or PIL Image objects
- Automatic resizing for model requirements
- Comprehensive guide in `REFERENCE_IMAGE.md`

## Configuration Updates

### Key Changes in `config.yaml`

1. **I2V Method Selection**:
   ```yaml
   i2v:
     method: "svd"  # or "wan"
   ```

2. **SVD Configuration** (Recommended):
   ```yaml
   i2v:
     svd:
       model_path: "stabilityai/stable-video-diffusion-img2vid"
       num_frames: 14  # or 25 for extended version
       motion_bucket_id: 127  # 1-255, higher = more motion
   ```

3. **LoRA Path Flexibility**:
   ```yaml
   sdxl:
     lora_path: "/path/to/lora.safetensors"  # Local file
     # OR
     lora_path: "username/lora-name"  # HuggingFace ID
   ```

## Dependencies Updated

- Added `huggingface-hub` for model downloads
- Added `einops` for tensor operations
- Updated `diffusers` requirement (>=0.21.0) for SVD support
- Real-ESRGAN dependencies properly configured

## Usage

### Basic Usage (Reference Image)
```bash
python main.py reference_image.jpg "your prompt here"
```

### With LoRA
1. Download LoRA or use HuggingFace ID
2. Update `config.yaml`:
   ```yaml
   sdxl:
     lora_path: "username/lora-name"
   ```
3. Run pipeline as normal

### Download Models
```bash
# Download all models
python setup_models.py --all

# Download specific models
python setup_models.py --sdxl --svd
python setup_models.py --lora username/lora-name
```

## File Structure

```
project_mask_two/
├── config.yaml              # Main configuration
├── main.py                  # CLI entry point
├── setup_models.py          # Model download helper
├── example.py               # Python API example
├── requirements.txt         # Dependencies
├── README.md                # Full documentation
├── QUICKSTART.md            # Quick reference
├── SETUP.md                 # Detailed setup
├── REFERENCE_IMAGE.md       # Reference image guide
└── src/
    ├── pipeline.py          # Main pipeline orchestrator
    ├── sdxl_img2img.py      # SDXL + LoRA img2img
    ├── image_upscaler.py    # Real-ESRGAN upscaling
    ├── i2v.py               # SVD/Wan2.2 I2V conversion
    └── video_upscaler.py    # Video frame upscaling
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure models**: Edit `config.yaml` (LoRA optional)
3. **Test with reference image**: `python main.py image.jpg "prompt"`
4. **Experiment**: Try different prompts, strengths, and settings

## Notes

- **SVD is recommended** over Wan2.2 for easier setup
- **LoRA is optional** - pipeline works without it
- **Models auto-download** on first use (requires internet)
- **Reference images** should be high quality for best results

