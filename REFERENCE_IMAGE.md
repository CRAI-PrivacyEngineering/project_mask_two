# Reference Image Guide

## Overview

The pipeline accepts a reference image as input, which serves as the starting point for the entire video generation process. This image goes through:

1. **SDXL img2img transformation** - Stylized/modified based on your prompt
2. **Image upscaling** - Enhanced resolution
3. **Image-to-video conversion** - Animated into video frames
4. **Video upscaling** - Final quality enhancement

## Image Requirements

### Recommended Specifications

- **Format**: PNG, JPEG, or any format supported by PIL
- **Resolution**: 
  - Minimum: 512x512 pixels
  - Recommended: 1024x1024 or higher
  - Optimal: 1024x576 (matches SVD input requirements)
- **Aspect Ratio**: 
  - For SVD: 16:9 (1024x576) works best
  - Other ratios will be resized/cropped as needed
- **Quality**: High-quality images produce better results

### Image Preparation Tips

1. **Use high-resolution images** - Better input = better output
2. **Ensure good lighting** - Well-lit images work better
3. **Avoid extreme crops** - Leave some context around subjects
4. **Consider composition** - Think about how motion will work
5. **Match aspect ratio** - 16:9 for SVD, or crop/resize accordingly

## Usage Examples

### Command Line

```bash
# Basic usage
python main.py reference_image.jpg "your prompt here"

# With negative prompt
python main.py reference_image.jpg "beautiful landscape" --negative-prompt "blurry, low quality"

# Custom output name
python main.py reference_image.jpg "your prompt" --output-name my_video
```

### Python API

```python
from src.pipeline import VideoGenerationPipeline

pipeline = VideoGenerationPipeline(config_path="config.yaml")

# Using image path
video_path = pipeline.generate(
    input_image="path/to/reference_image.jpg",
    prompt="a cinematic scene with dramatic lighting",
    negative_prompt="blurry, distorted, artifacts"
)

# Using PIL Image
from PIL import Image

image = Image.open("reference_image.jpg")
video_path = pipeline.generate(
    input_image=image,
    prompt="your prompt here"
)
```

## Image Processing Pipeline

### Step 1: SDXL img2img
- Your reference image is transformed based on the prompt
- `strength` parameter controls how much it changes (0.0-1.0)
  - Lower (0.3-0.5): More faithful to original
  - Higher (0.7-0.9): More transformation

### Step 2: Image Upscaling
- The transformed image is upscaled (default: 4x)
- Uses Real-ESRGAN for high-quality upscaling
- Output resolution depends on input and scale factor

### Step 3: Image-to-Video
- Upscaled image is converted to video frames
- SVD automatically resizes to 1024x576 if needed
- Motion is controlled by `motion_bucket_id` parameter

### Step 4: Video Upscaling
- Video frames are upscaled again for final quality
- Frame interpolation adds smoothness

## Best Practices

1. **Start with quality images** - Don't expect miracles from low-res inputs
2. **Experiment with strength** - Try different values (0.5, 0.75, 0.9)
3. **Use descriptive prompts** - More detail helps guide transformation
4. **Consider motion** - Think about what should move in the video
5. **Test with small batches** - Generate a few frames first to test settings

## Troubleshooting

### Image too small
- Upscale your reference image before using it
- Or reduce the upscaling factor in config

### Poor quality output
- Use higher resolution reference images
- Increase `num_inference_steps` in config
- Try different `strength` values

### Wrong aspect ratio
- Crop/resize your image to match desired output
- SVD works best with 16:9 (1024x576)

### Memory issues
- Reduce input image resolution
- Use `float32` instead of `float16` (slower but more stable)
- Reduce `num_frames` in I2V config

