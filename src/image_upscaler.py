"""
Image Upscaling Module
Supports Real-ESRGAN and Topaz-like upscaling methods
"""

import torch
from PIL import Image
import numpy as np
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not available. Install with: pip install realesrgan")


class ImageUpscaler:
    """Image upscaling using Real-ESRGAN or other methods"""
    
    def __init__(
        self,
        method: str = "realesrgan",
        scale_factor: int = 4,
        model_name: str = "RealESRGAN_x4plus",
        device: str = "cuda"
    ):
        """
        Initialize image upscaler
        
        Args:
            method: Upscaling method ("realesrgan" or "topaz")
            scale_factor: Upscaling factor (2, 4, etc.)
            model_name: Model name for Real-ESRGAN
            device: Device to run on
        """
        self.method = method
        self.scale_factor = scale_factor
        self.device = device
        
        if method == "realesrgan":
            if not REALESRGAN_AVAILABLE:
                raise ImportError(
                    "Real-ESRGAN not available. Install with: pip install realesrgan"
                )
            self._init_realesrgan(model_name)
        elif method == "topaz":
            logger.warning("Topaz API integration not implemented. Using Real-ESRGAN fallback.")
            if REALESRGAN_AVAILABLE:
                self._init_realesrgan(model_name)
                self.method = "realesrgan"
            else:
                raise ValueError("Topaz not available and Real-ESRGAN fallback failed")
        else:
            raise ValueError(f"Unknown upscaling method: {method}")
    
    def _init_realesrgan(self, model_name: str):
        """Initialize Real-ESRGAN upscaler"""
        # Map model names to actual model files
        model_mapping = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus_anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        }
        
        # Determine model architecture based on scale factor
        if self.scale_factor == 4:
            model = RRDBNet(num_in_block=5, num_out_block=4, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
        elif self.scale_factor == 2:
            model = RRDBNet(num_in_block=5, num_out_block=4, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=2)
        else:
            raise ValueError(f"Unsupported scale factor: {self.scale_factor}. Supported: 2, 4")
        
        model_url = model_mapping.get(model_name, model_mapping["RealESRGAN_x4plus"])
        
        # Initialize upsampler
        # RealESRGANer will download the model automatically if not present
        self.upsampler = RealESRGANer(
            scale=self.scale_factor,
            model_path=model_url,
            model=model,
            tile=0,  # Set to 0 for no tiling, increase if OOM
            tile_pad=10,
            pre_pad=0,
            half=self.device == "cuda" and torch.cuda.is_available()
        )
        
        logger.info(f"Real-ESRGAN initialized with {model_name} (scale: {self.scale_factor}x)")
    
    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscale an image
        
        Args:
            image: Input PIL Image
            
        Returns:
            Upscaled PIL Image
        """
        if self.method == "realesrgan":
            return self._upscale_realesrgan(image)
        elif self.method == "topaz":
            # Placeholder for Topaz API integration
            return self._upscale_realesrgan(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _upscale_realesrgan(self, image: Image.Image) -> Image.Image:
        """Upscale using Real-ESRGAN"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Upscale
        logger.info(f"Upscaling image from {img_array.shape[:2]}...")
        output, _ = self.upsampler.enhance(img_array, outscale=self.scale_factor)
        
        # Convert back to PIL Image
        upscaled_image = Image.fromarray(output)
        
        logger.info(f"Upscaled to {upscaled_image.size}")
        return upscaled_image
    
    def _upscale_topaz(self, image: Image.Image) -> Image.Image:
        """
        Upscale using Topaz API (placeholder)
        This would integrate with Topaz Video Enhance AI API if available
        """
        # TODO: Implement Topaz API integration
        raise NotImplementedError("Topaz API integration not yet implemented")

