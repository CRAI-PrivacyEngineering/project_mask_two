#!/usr/bin/env python3
"""
Main entry point for AI Video Generation Tool
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import logging

from src.pipeline import VideoGenerationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic AI videos from images using SDXL LoRA, img2img, upscaling, and I2V (SVD/Wan2.2)"
    )
    
    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input image file'
    )
    
    parser.add_argument(
        'prompt',
        type=str,
        help='Text prompt for image generation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default='',
        help='Negative prompt'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Output name for generated files (without extension)'
    )
    
    parser.add_argument(
        '--no-save-intermediate',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    args = parser.parse_args()
    
    # Validate input image
    if not Path(args.input_image).exists():
        logger.error(f"Input image not found: {args.input_image}")
        sys.exit(1)
    
    # Validate config file
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = VideoGenerationPipeline(config_path=args.config)
        
        # Generate video
        output_path = pipeline.generate(
            input_image=args.input_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_name=args.output_name,
            save_intermediate=not args.no_save_intermediate
        )
        
        logger.info(f"\n✓ Success! Video generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during video generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

