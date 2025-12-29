"""
Image-to-Video Module
Supports Stable Video Diffusion (SVD) and Wan2.2 I2V models
"""

import torch
from PIL import Image
import numpy as np
from typing import Optional, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from diffusers import StableVideoDiffusionPipeline
    SVD_AVAILABLE = True
except ImportError:
    SVD_AVAILABLE = False
    logger.warning("Stable Video Diffusion not available. Install diffusers>=0.21.0")


class ImageToVideo:
    """Image-to-Video converter supporting SVD and Wan2.2"""
    
    def __init__(
        self,
        method: str = "svd",
        model_path: str = "",
        device: str = "cuda",
        num_frames: int = 14,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 2,
        **kwargs
    ):
        """
        Initialize Image-to-Video model
        
        Args:
            method: Method to use ("svd" or "wan")
            model_path: Path to model or HuggingFace model ID
            device: Device to run on
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_bucket_id: Motion amount (1-255)
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Chunk size for decoding (memory efficiency)
            **kwargs: Additional method-specific parameters
        """
        self.method = method.lower()
        self.device = device
        self.num_frames = num_frames
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.noise_aug_strength = noise_aug_strength
        self.decode_chunk_size = decode_chunk_size
        self.kwargs = kwargs
        
        if self.method == "svd":
            self._init_svd(model_path)
        elif self.method == "wan":
            self._init_wan(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown I2V method: {method}. Choose 'svd' or 'wan'")
    
    def _init_svd(self, model_path: str):
        """Initialize Stable Video Diffusion"""
        if not SVD_AVAILABLE:
            raise ImportError(
                "Stable Video Diffusion requires diffusers>=0.21.0. "
                "Install with: pip install diffusers>=0.21.0"
            )
        
        # Default to HuggingFace model if no path provided
        if not model_path:
            model_path = "stabilityai/stable-video-diffusion-img2vid"
            logger.info(f"No model path provided, using default: {model_path}")
        
        logger.info(f"Loading Stable Video Diffusion from {model_path}...")
        
        # Determine dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load SVD pipeline
        self.model = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        # Enable memory optimizations
        try:
            self.model.enable_xformers_memory_efficient_attention()
        except:
            logger.warning("xformers not available, using default attention")
        
        # Enable model CPU offload for memory efficiency
        self.model.enable_model_cpu_offload()
        
        logger.info("Stable Video Diffusion loaded successfully")
    
    def _init_wan(self, model_path: str, **kwargs):
        """Initialize Wan2.2 I2V"""
        if not model_path:
            logger.warning(
                "Wan2.2 model path not provided. "
                "Please set model_path in config.yaml or provide it during initialization."
            )
            logger.info(
                "Wan2.2 requires custom setup. See: https://github.com/Wan-Video/Wan2.2"
            )
            self.model = None
            return
        
        logger.info(f"Loading Wan2.2 model from {model_path}...")
        
        # Wan2.2 uses a custom repository structure
        # This is a placeholder - you'll need to integrate based on Wan2.2's actual API
        # Check: https://github.com/Wan-Video/Wan2.2 for implementation details
        
        if os.path.exists(model_path):
            # TODO: Implement actual Wan2.2 loading
            # Example structure (needs to be adapted to actual Wan2.2 API):
            # from wan2 import Wan2Pipeline  # Hypothetical
            # self.model = Wan2Pipeline.from_pretrained(model_path)
            # self.model = self.model.to(self.device)
            logger.warning("Wan2.2 integration not fully implemented. Using placeholder.")
            self.model = None
        else:
            logger.error(f"Wan2.2 model not found at {model_path}")
            self.model = None
    
    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate video frames from input image
        
        Args:
            image: Input PIL Image (will be resized to model requirements)
            prompt: Optional text prompt (not used by SVD, but kept for compatibility)
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of PIL Images (video frames)
        """
        if self.method == "svd":
            return self._generate_svd(image, seed=seed, **kwargs)
        elif self.method == "wan":
            return self._generate_wan(image, prompt=prompt, seed=seed, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_svd(self, image: Image.Image, seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        """Generate frames using Stable Video Diffusion"""
        if self.model is None:
            raise RuntimeError("SVD model not initialized")
        
        # SVD requires specific image size (1024x576)
        # Resize image while maintaining aspect ratio
        target_size = (1024, 576)
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        from diffusers.utils import load_image
        import torch
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating {self.num_frames} frames with SVD...")
        
        # Generate video frames
        output = self.model(
            image=image_resized,
            decode_chunk_size=self.decode_chunk_size,
            num_frames=self.num_frames,
            motion_bucket_id=self.motion_bucket_id,
            noise_aug_strength=self.noise_aug_strength,
            generator=generator,
            **kwargs
        )
        
        # SVD returns frames in a list format
        frames = output.frames[0] if hasattr(output, 'frames') else output
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Convert tensor to numpy array
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().numpy()
                # Normalize from [-1, 1] to [0, 255] if needed
                if frame_np.min() < 0:
                    frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                else:
                    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                
                # Handle different tensor formats
                if len(frame_np.shape) == 3:
                    if frame_np.shape[0] == 3:  # CHW format
                        frame_np = frame_np.transpose(1, 2, 0)  # HWC format
                    elif frame_np.shape[2] == 3:  # Already HWC
                        pass
                    else:
                        raise ValueError(f"Unexpected frame shape: {frame_np.shape}")
                
                pil_frames.append(Image.fromarray(frame_np))
            elif isinstance(frame, Image.Image):
                pil_frames.append(frame)
            else:
                # Assume it's already a numpy array
                pil_frames.append(Image.fromarray(frame))
        
        logger.info(f"Generated {len(pil_frames)} frames")
        return pil_frames
    
    def _generate_wan(self, image: Image.Image, prompt: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        """Generate frames using Wan2.2"""
        if self.model is None:
            logger.warning("Wan2.2 model not loaded. Generating placeholder frames.")
            return self._generate_placeholder_frames(image)
        
        # TODO: Implement actual Wan2.2 generation
        # This depends on Wan2.2's actual API
        # Example structure:
        # frames = self.model.generate(
        #     image=image,
        #     prompt=prompt,
        #     num_frames=self.num_frames,
        #     motion_bucket_id=self.motion_bucket_id,
        #     generator=generator,
        #     **kwargs
        # )
        
        logger.warning("Wan2.2 generation not fully implemented. Using placeholder.")
        return self._generate_placeholder_frames(image)
    
    def _generate_placeholder_frames(self, image: Image.Image) -> List[Image.Image]:
        """Generate placeholder frames (for testing without model)"""
        frames = []
        for i in range(self.num_frames):
            frames.append(image.copy())
        return frames

