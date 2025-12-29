"""
Main Pipeline for AI Video Generation
Orchestrates the entire workflow: SDXL img2img -> Image Upscale -> Wan I2V -> Video Upscale
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import os

from .sdxl_img2img import SDXLLoRAImg2Img
from .image_upscaler import ImageUpscaler
from .i2v import ImageToVideo
from .video_upscaler import VideoUpscaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoGenerationPipeline:
    """Complete pipeline for generating high-quality AI videos"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # SDXL img2img
        sdxl_config = self.config.get('sdxl', {})
        self.img2img = SDXLLoRAImg2Img(
            model_path=sdxl_config.get('model_path', 'stabilityai/stable-diffusion-xl-base-1.0'),
            lora_path=sdxl_config.get('lora_path'),
            lora_scale=sdxl_config.get('lora_scale', 1.0),
            device=sdxl_config.get('device', 'cuda'),
            dtype=sdxl_config.get('dtype', 'float16')
        )
        
        # Image upscaler
        upscale_config = self.config.get('image_upscale', {})
        self.image_upscaler = ImageUpscaler(
            method=upscale_config.get('method', 'realesrgan'),
            scale_factor=upscale_config.get('scale_factor', 4),
            model_name=upscale_config.get('model_name', 'RealESRGAN_x4plus'),
            device=sdxl_config.get('device', 'cuda')
        )
        
        # Image-to-Video (SVD or Wan2.2)
        i2v_config = self.config.get('i2v', {})
        i2v_method = i2v_config.get('method', 'svd')
        
        if i2v_method == 'svd':
            svd_config = i2v_config.get('svd', {})
            self.i2v = ImageToVideo(
                method='svd',
                model_path=svd_config.get('model_path', ''),
                device=sdxl_config.get('device', 'cuda'),
                num_frames=svd_config.get('num_frames', 14),
                fps=svd_config.get('fps', 7),
                motion_bucket_id=svd_config.get('motion_bucket_id', 127),
                noise_aug_strength=svd_config.get('noise_aug_strength', 0.02),
                decode_chunk_size=svd_config.get('decode_chunk_size', 2)
            )
        else:  # wan
            wan_config = i2v_config.get('wan', {})
            self.i2v = ImageToVideo(
                method='wan',
                model_path=wan_config.get('model_path', ''),
                device=sdxl_config.get('device', 'cuda'),
                num_frames=wan_config.get('num_frames', 16),
                fps=wan_config.get('fps', 8),
                motion_bucket_id=wan_config.get('motion_bucket_id', 127),
                noise_aug_strength=wan_config.get('cond_aug', 0.02),
                size=wan_config.get('size', '1280*720'),
                offload_model=wan_config.get('offload_model', True),
                convert_model_dtype=wan_config.get('convert_model_dtype', True)
            )
        
        # Video upscaler
        video_upscale_config = self.config.get('video_upscale', {})
        self.video_upscaler = VideoUpscaler(
            method=video_upscale_config.get('method', 'realesrgan'),
            scale_factor=video_upscale_config.get('scale_factor', 2),
            frame_interpolation=video_upscale_config.get('frame_interpolation', True),
            device=sdxl_config.get('device', 'cuda')
        )
        
        # Output configuration
        self.output_config = self.config.get('output', {})
        self.output_dir = Path(self.output_config.get('base_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline initialized successfully")
    
    def generate(
        self,
        input_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        output_name: Optional[str] = None,
        save_intermediate: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate video from input image
        
        Args:
            input_image: Path to input image or PIL Image
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            output_name: Name for output files (without extension)
            save_intermediate: Whether to save intermediate results
            **kwargs: Additional generation parameters
            
        Returns:
            Path to generated video file
        """
        save_intermediate = save_intermediate if save_intermediate is not None else self.output_config.get('save_intermediate', True)
        
        # Generate unique output name if not provided
        if output_name is None:
            from datetime import datetime
            output_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("=" * 60)
        logger.info("Starting video generation pipeline")
        logger.info("=" * 60)
        
        # Step 1: SDXL img2img
        logger.info("\n[Step 1/4] SDXL Image-to-Image Generation")
        logger.info("-" * 60)
        img2img_config = self.config.get('img2img', {})
        generated_image = self.img2img.generate(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=img2img_config.get('strength', 0.75),
            guidance_scale=img2img_config.get('guidance_scale', 7.5),
            num_inference_steps=img2img_config.get('num_inference_steps', 50),
            seed=img2img_config.get('seed'),
            **kwargs
        )
        
        if save_intermediate:
            img2img_path = self.output_dir / f"{output_name}_step1_img2img.png"
            generated_image.save(img2img_path)
            logger.info(f"Saved img2img result: {img2img_path}")
        
        # Step 2: Image upscaling
        logger.info("\n[Step 2/4] Image Upscaling")
        logger.info("-" * 60)
        upscaled_image = self.image_upscaler.upscale(generated_image)
        
        if save_intermediate:
            upscale_path = self.output_dir / f"{output_name}_step2_upscaled.png"
            upscaled_image.save(upscale_path)
            logger.info(f"Saved upscaled image: {upscale_path}")
        
        # Step 3: Image-to-Video conversion
        i2v_method = self.config.get('i2v', {}).get('method', 'svd')
        logger.info(f"\n[Step 3/4] Image-to-Video Conversion ({i2v_method.upper()})")
        logger.info("-" * 60)
        video_frames = self.i2v.generate(
            image=upscaled_image,
            prompt=prompt,
            seed=img2img_config.get('seed'),
            **kwargs
        )
        
        if save_intermediate:
            frames_dir = self.output_dir / f"{output_name}_step3_frames"
            frames_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(video_frames):
                frame.save(frames_dir / f"frame_{i:04d}.png")
            logger.info(f"Saved video frames: {frames_dir}")
        
        # Step 4: Video upscaling
        logger.info("\n[Step 4/4] Video Upscaling")
        logger.info("-" * 60)
        final_frames = self.video_upscaler.upscale_frames(video_frames)
        
        # Save final video
        video_config = self.output_config
        video_path = self.output_dir / f"{output_name}_final.{video_config.get('video_format', 'mp4')}"
        
        self.video_upscaler.frames_to_video(
            frames=final_frames,
            output_path=str(video_path),
            fps=self.i2v.fps,
            codec=video_config.get('video_codec', 'libx264'),
            bitrate=video_config.get('video_bitrate', '10M')
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final video saved: {video_path}")
        logger.info("=" * 60)
        
        return str(video_path)

