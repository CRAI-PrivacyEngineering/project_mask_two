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
        
        # Get frame dimensions
        width, height = frames[0].size
        
        # Create video writer
        # Convert codec name to fourcc format
        codec_map = {
            'libx264': 'mp4v',
            'libx265': 'HEVC',
            'mjpeg': 'MJPG',
            'xvid': 'XVID'
        }
        fourcc_code = codec_map.get(codec.lower(), 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        
        logger.info(f"Saving video to {output_path} ({width}x{height}, {fps} fps)...")
        
        for frame in frames:
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        
        out.release()
        logger.info("Video saved successfully")

