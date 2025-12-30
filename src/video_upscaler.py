"""
Video Upscaling Module
Upscales video frames using Real-ESRGAN or other methods
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Optional
import logging
import cv2
from pathlib import Path

from .image_upscaler import ImageUpscaler

logger = logging.getLogger(__name__)


class VideoUpscaler:
    """Video upscaling using frame-by-frame upscaling"""
    
    def __init__(
        self,
        method: str = "realesrgan",
        scale_factor: int = 2,
        frame_interpolation: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize video upscaler
        
        Args:
            method: Upscaling method
            scale_factor: Upscaling factor
            frame_interpolation: Whether to interpolate frames for smoother motion
            device: Device to run on
        """
        self.method = method
        self.scale_factor = scale_factor
        self.frame_interpolation = frame_interpolation
        self.device = device
        
        # Use image upscaler for frame-by-frame processing
        self.image_upscaler = ImageUpscaler(
            method=method,
            scale_factor=scale_factor,
            device=device
        )
        
        logger.info(f"Video upscaler initialized (method: {method}, scale: {scale_factor}x)")
    
    def upscale_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        Upscale video frames
        
        Args:
            frames: List of PIL Images (video frames)
            
        Returns:
            List of upscaled PIL Images
        """
        logger.info(f"Upscaling {len(frames)} frames...")
        
        upscaled_frames = []
        for i, frame in enumerate(frames):
            logger.info(f"Upscaling frame {i+1}/{len(frames)}")
            upscaled_frame = self.image_upscaler.upscale(frame)
            upscaled_frames.append(upscaled_frame)
        
        # Optional: Frame interpolation for smoother motion
        if self.frame_interpolation and len(upscaled_frames) > 1:
            logger.info("Applying frame interpolation...")
            upscaled_frames = self._interpolate_frames(upscaled_frames)
        
        return upscaled_frames
    
    def _interpolate_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        Interpolate frames for smoother motion
        Uses simple blending between frames
        """
        if len(frames) < 2:
            return frames
        
        interpolated = [frames[0]]  # Keep first frame
        
        for i in range(len(frames) - 1):
            # Add original frame
            interpolated.append(frames[i + 1])
            
            # Optional: Add interpolated frame between current and next
            # This creates smoother transitions
            if self.frame_interpolation:
                current = np.array(frames[i])
                next_frame = np.array(frames[i + 1])
                
                # Blend frames
                blended = ((current.astype(np.float32) + next_frame.astype(np.float32)) / 2).astype(np.uint8)
                interpolated.insert(-1, Image.fromarray(blended))
        
        return interpolated
    
    def frames_to_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 24,
        codec: str = "libx264",
        bitrate: str = "10M"
    ):
        """
        Save frames as video file
        
        Args:
            frames: List of PIL Images
            output_path: Output video file path
            fps: Frames per second
            codec: Video codec
            bitrate: Video bitrate
        """
        if not frames:
            raise ValueError("No frames to save")
        
        # Use imageio-ffmpeg for proper H.264 encoding
        try:
            import imageio
            import imageio_ffmpeg
            
            logger.info(f"Saving video to {output_path} ({frames[0].size[0]}x{frames[0].size[1]}, {fps} fps)...")
            
            # Convert frames to numpy arrays
            frame_arrays = [np.array(frame) for frame in frames]
            
            # Write video using imageio with proper codec
            # Use ffmpeg plugin for better compatibility
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                quality=8,  # High quality (0-10, higher is better)
                pixelformat='yuv420p',  # Ensures compatibility
                macro_block_size=None,  # Auto-detect
                ffmpeg_params=['-preset', 'slow', '-crf', '18']  # High quality encoding
            )
            
            for frame_array in frame_arrays:
                writer.append_data(frame_array)
            
            writer.close()
            logger.info("Video saved successfully")
            
        except Exception as e:
            logger.warning(f"imageio-ffmpeg failed: {e}, falling back to OpenCV")
            # Fallback to OpenCV
            width, height = frames[0].size
            
            # Use H.264 codec properly
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )
            
            if not out.isOpened():
                # Try alternative codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(frame_cv)
            
            out.release()
            logger.info("Video saved successfully (using OpenCV fallback)")

