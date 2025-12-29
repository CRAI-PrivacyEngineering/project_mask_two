# AI Video Generation Pipeline

A comprehensive tool for generating realistic, high-quality AI videos from images using:
- **SDXL LoRA** for image-to-image generation
- **Image upscaling** (Real-ESRGAN or Topaz)
- **Wan I2V** for image-to-video conversion
- **Video upscaling** for final quality enhancement

## Features

- 🎨 SDXL-based image-to-image generation with LoRA support
- 🔍 High-quality image upscaling (4x or higher)
- 🎬 Image-to-video conversion with motion control
- 📹 Video upscaling with frame interpolation
- 💾 Intermediate result saving for debugging
- ⚙️ Fully configurable pipeline

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended)
- 20GB+ free disk space for models
- NVIDIA GPU with 8GB+ VRAM (recommended for best performance)

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch** (choose based on your system):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Download models** (optional - models download automatically on first use):
```bash
# Download all models
python setup_models.py --all

# Or download individually
python setup_models.py --sdxl --svd
python setup_models.py --lora username/lora-name  # If you have a LoRA on HuggingFace
```

**Note**: Models will be downloaded automatically on first use if not pre-downloaded.

## Configuration

Edit `config.yaml` to configure the pipeline:

### SDXL Configuration
```yaml
sdxl:
  model_path: "stabilityai/stable-diffusion-xl-base-1.0"  # HuggingFace ID or local path
  lora_path: "/path/to/your/lora.safetensors"  # Or HuggingFace LoRA ID
  # Examples:
  # lora_path: "/path/to/lora.safetensors"
  # lora_path: "username/lora-name"  # HuggingFace ID
  lora_scale: 1.0  # LoRA strength (0.0-2.0)
  device: "cuda"  # or "cpu"
  dtype: "float16"  # or "float32"
```

### Image-to-Image Settings
```yaml
img2img:
  strength: 0.75  # How much to transform (0.0-1.0). Lower = more faithful to original
  guidance_scale: 7.5  # How closely to follow prompt (1-20)
  num_inference_steps: 50  # More steps = better quality but slower
  seed: null  # Set to a number for reproducibility
```

### Upscaling Settings
```yaml
image_upscale:
  method: "realesrgan"  # Options: "realesrgan", "topaz"
  scale_factor: 4  # 2x, 4x, etc.
  model_name: "RealESRGAN_x4plus"  # or "RealESRGAN_x4plus_anime"
```

### Image-to-Video Settings
```yaml
i2v:
  method: "svd"  # Options: "svd" (recommended), "wan"
  
  # Stable Video Diffusion (SVD) - Recommended
  svd:
    model_path: "stabilityai/stable-video-diffusion-img2vid"  # HuggingFace ID
    num_frames: 14  # SVD supports 14 or 25 frames
    fps: 7
    motion_bucket_id: 127  # Motion amount (1-255, higher = more motion)
    noise_aug_strength: 0.02
    decode_chunk_size: 2  # Memory efficiency
    
  # Wan2.2 (requires custom setup)
  wan:
    model_path: "/path/to/Wan2.2-I2V-A14B"  # Update this
    num_frames: 16
    fps: 8
    motion_bucket_id: 127
```

## Usage

### Basic Usage

```bash
python main.py input_image.jpg "your prompt here"
```

### Advanced Usage

```bash
python main.py input_image.jpg "your prompt here" \
  --config custom_config.yaml \
  --negative-prompt "blurry, low quality" \
  --output-name my_video \
  --no-save-intermediate
```

### Python API

```python
from src.pipeline import VideoGenerationPipeline

# Initialize pipeline
pipeline = VideoGenerationPipeline(config_path="config.yaml")

# Generate video
video_path = pipeline.generate(
    input_image="input.jpg",
    prompt="a beautiful landscape",
    negative_prompt="blurry, distorted",
    output_name="my_video"
)

print(f"Video saved to: {video_path}")
```

## Pipeline Workflow

1. **SDXL img2img**: Transforms input image using SDXL with LoRA weights
2. **Image Upscaling**: Upscales the generated image (4x by default)
3. **Wan I2V**: Converts upscaled image to video frames
4. **Video Upscaling**: Upscales video frames and applies frame interpolation

## Output Structure

```
outputs/
├── video_20240101_120000_step1_img2img.png
├── video_20240101_120000_step2_upscaled.png
├── video_20240101_120000_step3_frames/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
└── video_20240101_120000_final.mp4
```

## Model Requirements

### SDXL Base Model
- Automatically downloaded from HuggingFace on first use
- Model ID: `stabilityai/stable-diffusion-xl-base-1.0`

### SDXL LoRA
- Download LoRA weights in `.safetensors` format or use HuggingFace LoRA ID
- Update `lora_path` in `config.yaml`
- Examples:
  - Local file: `/path/to/lora.safetensors`
  - HuggingFace: `username/lora-name`

### Stable Video Diffusion (SVD) - Recommended
- Automatically downloaded from HuggingFace on first use
- Model ID: `stabilityai/stable-video-diffusion-img2vid`
- Alternative: `stabilityai/stable-video-diffusion-img2vid-xt` (extended version)

### Wan2.2 I2V (Advanced)
- Requires custom setup from [Wan2.2 repository](https://github.com/Wan-Video/Wan2.2)
- Download model weights and update `i2v.wan.model_path` in `config.yaml`
- Note: Full integration requires adapting to Wan2.2's API

### Real-ESRGAN
- Models are downloaded automatically on first use
- Available models:
  - `RealESRGAN_x4plus`: General purpose (recommended)
  - `RealESRGAN_x4plus_anime`: Optimized for anime/illustrations

## Tips for Best Results

1. **Consistency**: Use the same seed across runs for consistent results
2. **Quality**: Higher `num_inference_steps` improves quality but increases generation time
3. **Motion**: Adjust `motion_bucket_id` in Wan I2V config to control motion amount
4. **Upscaling**: Use 4x image upscaling for best quality before video conversion
5. **Memory**: Enable VAE slicing and use `float16` to reduce memory usage

## Troubleshooting

### Out of Memory Errors
- Reduce `num_inference_steps`
- Use `float16` dtype
- Enable VAE slicing (already enabled by default)
- Reduce image resolution before processing

### Model Not Found
- Check model paths in `config.yaml`
- Ensure LoRA weights are in `.safetensors` format or use HuggingFace ID
- For SVD: Models download automatically, check internet connection
- For Wan2.2: Verify model path is correct and model is properly set up

### Slow Generation
- Use GPU (`device: "cuda"`)
- Install `xformers` for faster attention
- Reduce `num_inference_steps` or `num_frames`

## License

This project uses various open-source models and libraries. Please check individual licenses:
- SDXL: [Stability AI License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- Real-ESRGAN: BSD 3-Clause License
- Wan I2V: Check original repository license

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Stability AI for SDXL
- Real-ESRGAN team for upscaling models
- Wan I2V developers

