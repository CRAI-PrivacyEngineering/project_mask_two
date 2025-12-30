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

- Python 3.10+ (Python 3.14.2 tested and working)
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended)
- 20GB+ free disk space for models
- NVIDIA GPU with 8GB+ VRAM (recommended for best performance)
- CUDA 11.8, 12.1, or 13.0 (CUDA 13.0 tested and working)

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch** (choose based on your CUDA version):
```bash
# For CUDA 13.0 (newest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

4. **Install dependencies**:
```bash
pip install --upgrade pip
pip install diffusers>=0.21.0 transformers>=4.30.0 accelerate>=0.20.0 safetensors>=0.3.0
pip install pillow>=9.5.0 opencv-python>=4.7.0 numpy>=1.24.0 pyyaml>=6.0 tqdm>=4.65.0
pip install imageio>=2.28.0 imageio-ffmpeg>=0.4.8 huggingface-hub>=0.16.0 einops>=0.6.0
pip install --no-deps realesrgan>=0.3.0

# Install basicsr (may require manual fix for Python 3.14+)
# If installation fails, see Troubleshooting section below
pip install basicsr>=1.4.2

# Install Real-ESRGAN dependencies
pip install facexlib>=0.2.5 gfpgan>=1.3.5

# Install YouTube downloader (for YouTube URL support)
pip install yt-dlp

# Optional: xformers (may not be available for CUDA 13.0)
# pip install xformers>=0.0.20
```

**Note**: For YouTube URL support, you also need `ffmpeg` installed on your system:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Note**: If `basicsr` installation fails on Python 3.14+, you may need to install it manually:
```bash
cd /tmp
git clone --depth 1 https://github.com/xinntao/BasicSR.git basicsr_fix
cd basicsr_fix
# Fix setup.py version issue (see Troubleshooting)
python3 -c "
import re
with open('setup.py', 'r') as f:
    content = f.read()
new_content = re.sub(
    r'def get_version\(\):.*?return locals\(\)\[\'__version__\'\]',
    '''def get_version():
    return '1.4.2\'''',
    content,
    flags=re.DOTALL
)
with open('setup.py', 'w') as f:
    f.write(new_content)
"
pip install .
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

**Important**: Always activate the virtual environment first:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then run:
```bash
python main.py input_image.jpg "your prompt here"
```

### Using YouTube Video URLs

You can provide a YouTube URL instead of an image file. The pipeline will automatically download the video and extract a frame:

```bash
# Basic usage with YouTube URL
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" "your prompt here"

# Extract frame from specific time (default is 00:00:01)
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" "your prompt" --frame-time "00:00:05"

# Short YouTube URL format
python main.py "https://youtu.be/VIDEO_ID" "your prompt"
```

**Requirements for YouTube URLs:**
- Install `yt-dlp`: `pip install yt-dlp`
- `ffmpeg` must be installed on your system
- The video must be publicly accessible (no private/unlisted videos without authentication)

**Example:**
```bash
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  "a person performing an action, natural movement, smooth motion" \
  --frame-time "00:00:10"
```

### Generating Videos with Action Prompts

To generate a video where a person in the image performs a specific action, use descriptive prompts:

```bash
# Example: Person waving (from image)
python main.py person_photo.jpg "a person waving their hand, smiling, natural movement, smooth motion"

# Example: Person waving (from YouTube)
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" "a person waving their hand, natural movement"

# Example: Person walking
python main.py person_photo.jpg "a person walking forward, natural gait, smooth leg movement, realistic motion"

# Example: Person turning head
python main.py person_photo.jpg "a person turning their head to the left, natural head movement, smooth transition"

# Example: Person raising arm
python main.py person_photo.jpg "a person raising their right arm up, natural arm movement, smooth motion"
```

**Tips for action prompts**:
- Be specific about the body part and movement direction
- Include words like "natural movement", "smooth motion", "realistic motion" for better results
- Describe the action clearly: "waving hand", "turning head", "walking forward", etc.
- Use negative prompts to avoid unwanted artifacts

### Advanced Usage

```bash
python main.py input_image.jpg "your prompt here" \
  --config custom_config.yaml \
  --negative-prompt "blurry, low quality, distorted, artifacts" \
  --output-name my_video \
  --no-save-intermediate
```

### Using Video Input (Local Video Files)

If you have a local video file and want to use a specific frame:
```bash
# Extract a frame first (using ffmpeg)
ffmpeg -i input_video.mp4 -vf "select=eq(n\,0)" -vsync vfr frame_0000.jpg

# Or extract frame at specific time
ffmpeg -ss 00:00:05 -i input_video.mp4 -vframes 1 -q:v 2 frame.jpg

# Then use the frame
python main.py frame_0000.jpg "person performing action, natural movement"
```

**Note**: For YouTube videos, you can use the URL directly (see "Using YouTube Video URLs" section above).

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

## Step-by-Step Guide: Generating Action Videos

### Prerequisites
- An image file (JPG, PNG) containing a person or subject
- The virtual environment activated (`source venv/bin/activate`)

### Step 1: Prepare Your Input

**Option A: Image File**
- Use a clear, well-lit image (JPG, PNG)
- Ensure the subject is clearly visible
- Recommended resolution: 512x512 or higher (will be upscaled automatically)

**Option B: YouTube URL**
- Provide a YouTube video URL
- The pipeline will extract a frame automatically
- Use `--frame-time` to specify which frame to extract (default: 00:00:01)
- Example: `python main.py "https://www.youtube.com/watch?v=VIDEO_ID" "your prompt" --frame-time "00:00:10"`

### Step 2: Craft Your Action Prompt
The prompt should describe:
1. **The subject**: "a person", "a woman", "a man", etc.
2. **The action**: Be specific about the movement
3. **Motion quality**: Add terms like "natural movement", "smooth motion"

**Good prompt examples**:
- `"a person waving their hand, natural arm movement, smooth motion, realistic"`
- `"a person walking forward, natural gait, smooth leg movement, realistic motion"`
- `"a person turning their head to the left, natural head movement, smooth transition"`
- `"a person raising their right arm up, natural arm movement, smooth motion"`

### Step 3: Run the Pipeline

```bash
# Basic command
python main.py your_image.jpg "your action prompt here"

# With negative prompt for better quality
python main.py your_image.jpg "a person waving their hand, natural movement" \
  --negative-prompt "blurry, low quality, distorted, artifacts, unnatural movement"

# With custom output name
python main.py your_image.jpg "a person walking forward, natural gait" \
  --output-name walking_person
```

### Step 4: Find Your Output
The generated video will be saved in the `outputs/` directory:
- `outputs/your_video_final.mp4` - Final video
- `outputs/your_video_step1_img2img.png` - After SDXL transformation
- `outputs/your_video_step2_upscaled.png` - After upscaling
- `outputs/your_video_step3_frames/` - Individual video frames

### Example: Complete Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run with action prompt
python main.py person_photo.jpg \
  "a person waving their hand enthusiastically, natural arm movement, smooth motion, realistic, high quality" \
  --negative-prompt "blurry, low quality, distorted, artifacts, unnatural movement, jittery"

# 3. Wait for processing (may take several minutes depending on GPU)

# 4. Check output
ls -lh outputs/
```

### Tips for Better Action Videos

1. **Be Specific**: Instead of "moving", say "waving hand" or "turning head"
2. **Direction Matters**: Specify "to the left", "forward", "upward", etc.
3. **Motion Quality**: Always include "natural movement" or "smooth motion"
4. **Negative Prompts**: Use to avoid common issues:
   - `"blurry, low quality, distorted, artifacts, unnatural movement, jittery, flickering"`
5. **Adjust Motion**: Edit `config.yaml` to change `motion_bucket_id` (1-255, higher = more motion)
6. **Consistency**: Use the same `seed` in config for reproducible results

## Pipeline Workflow

1. **SDXL img2img**: Transforms input image using SDXL with LoRA weights based on your prompt
2. **Image Upscaling**: Upscales the generated image (4x by default) using Real-ESRGAN
3. **Image-to-Video (SVD)**: Converts upscaled image to video frames using Stable Video Diffusion
4. **Video Upscaling**: Upscales video frames (2x by default) and applies frame interpolation for smoother motion

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
- Install `xformers` for faster attention (may not be available for CUDA 13.0 - code will use default attention)
- Reduce `num_inference_steps` or `num_frames`
- The pipeline will work without xformers, just slightly slower

### BasicsR Installation Issues (Python 3.14+)
If `basicsr` fails to install with a `KeyError: '__version__'` error:
1. Clone the repository manually (see Setup step 4)
2. Fix the `setup.py` file as shown in the installation instructions
3. Install from the fixed directory

### CUDA 13.0 Compatibility
- PyTorch 2.9.1+cu130 supports CUDA 13.0
- You may see a warning about compute capability 12.1 - this is expected for very new GPUs and can be ignored
- xformers may not be available for CUDA 13.0 - the pipeline will automatically use default attention

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

