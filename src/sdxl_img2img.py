"""
SDXL LoRA Image-to-Image Module
Handles loading SDXL models with LoRA weights and performing img2img generation
"""

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class SDXLLoRAImg2Img:
    """SDXL Image-to-Image pipeline with LoRA support"""
    
    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        device: str = "cuda",
        dtype: str = "float16"
    ):
        """
        Initialize SDXL img2img pipeline with LoRA
        
        Args:
            model_path: Path to SDXL base model or HuggingFace model ID
            lora_path: Path to LoRA weights file (.safetensors)
            lora_scale: Scale factor for LoRA weights
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type ("float16" or "float32")
        """
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        
        logger.info(f"Loading SDXL model from {model_path}...")
        
        # Load VAE separately for better quality
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype
        )
        
        # Initialize img2img pipeline
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if dtype == "float16" else None
        )
        
        # Load LoRA weights if provided
        if lora_path:
            logger.info(f"Loading LoRA weights from {lora_path}...")
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=lora_scale)
        
        # Enable memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            logger.warning("xformers not available, using default attention")
        
        # Move to device
        self.pipe = self.pipe.to(device)
        
        # Enable VAE slicing for memory efficiency
        self.pipe.vae.enable_slicing()
        
        logger.info("SDXL pipeline loaded successfully")
    
    def generate(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate image from input image using img2img
        
        Args:
            image: Input PIL Image or path to image file
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            strength: How much to transform the image (0.0-1.0)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for pipeline
            
        Returns:
            Generated PIL Image
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating image with prompt: {prompt[:50]}...")
        
        # Generate image
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs
        )
        
        return result.images[0]
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache() if self.device == "cuda" else None

