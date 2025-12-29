# Setup Guide

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure your models:**
   - Edit `config.yaml` and set your LoRA path
   - Set your Wan I2V model path (if available)
   - Adjust other settings as needed

3. **Run the pipeline:**
```bash
python main.py your_image.jpg "your prompt here"
```

## Detailed Setup

### 1. Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch

Install PyTorch with CUDA support (if you have an NVIDIA GPU):

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

#### SDXL Base Model
The SDXL base model will be downloaded automatically on first use from HuggingFace.

#### LoRA Weights
1. Download your SDXL LoRA weights (`.safetensors` format)
2. Update `config.yaml`:
```yaml
sdxl:
  lora_path: "/path/to/your/lora.safetensors"
```

#### Real-ESRGAN Models
Real-ESRGAN models are downloaded automatically on first use.

#### Wan I2V Model
**Important:** The Wan I2V integration requires the actual model files. Currently, the code includes a placeholder implementation.

To integrate Wan I2V:

1. **Download the Wan I2V model** from the official repository
2. **Update the `wan_i2v.py` file** to match the actual model API:
   - Check the model's documentation for loading instructions
   - Update the `_load_model()` method
   - Update the `generate()` method to use the actual inference API

3. **Update `config.yaml`:**
```yaml
wan_i2v:
  model_path: "/path/to/wan_i2v/model"
```

### 5. Verify Installation

Test the installation:

```bash
python -c "from src.pipeline import VideoGenerationPipeline; print('✓ Installation successful')"
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `num_inference_steps` in config
- Use `float16` dtype
- Reduce image resolution before processing
- Enable VAE slicing (already enabled by default)

### Real-ESRGAN Import Error

```bash
pip install basicsr
pip install realesrgan
```

### Model Download Issues

- Check your internet connection
- Some models may require HuggingFace login
- For large models, consider downloading manually and setting local paths

### Wan I2V Not Working

The current implementation includes a placeholder. You need to:
1. Obtain the Wan I2V model
2. Integrate it according to its API
3. Update `src/wan_i2v.py` with the correct implementation

## Next Steps

- Read `README.md` for usage instructions
- Check `example.py` for code examples
- Customize `config.yaml` for your needs

