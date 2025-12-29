#!/usr/bin/env python3
"""
Helper script to download and setup models
"""

import os
import sys
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface-hub")


def download_sdxl_model(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", cache_dir: str = None):
    """Download SDXL base model (will be cached by diffusers)"""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available")
        return False
    
    logger.info(f"Downloading SDXL model: {model_id}")
    logger.info("Note: This will be cached by diffusers library")
    
    try:
        # Diffusers will download automatically, but we can pre-download
        from diffusers import StableDiffusionXLImg2ImgPipeline
        logger.info("Pre-loading SDXL model (this may take a while)...")
        _ = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        logger.info("✓ SDXL model ready")
        return True
    except Exception as e:
        logger.error(f"Error downloading SDXL model: {e}")
        return False


def download_svd_model(model_id: str = "stabilityai/stable-video-diffusion-img2vid", cache_dir: str = None):
    """Download Stable Video Diffusion model"""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available")
        return False
    
    logger.info(f"Downloading SVD model: {model_id}")
    logger.info("Note: This will be cached by diffusers library")
    
    try:
        from diffusers import StableVideoDiffusionPipeline
        logger.info("Pre-loading SVD model (this may take a while)...")
        _ = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        logger.info("✓ SVD model ready")
        return True
    except Exception as e:
        logger.error(f"Error downloading SVD model: {e}")
        return False


def download_lora(lora_id: str, output_dir: str = "./models/lora"):
    """Download LoRA weights from HuggingFace"""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading LoRA: {lora_id}")
    
    try:
        snapshot_download(
            repo_id=lora_id,
            local_dir=str(output_path / lora_id.split("/")[-1]),
            local_dir_use_symlinks=False
        )
        logger.info(f"✓ LoRA downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading LoRA: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for AI Video Generation Pipeline")
    
    parser.add_argument(
        '--sdxl',
        action='store_true',
        help='Download SDXL base model'
    )
    
    parser.add_argument(
        '--svd',
        action='store_true',
        help='Download Stable Video Diffusion model'
    )
    
    parser.add_argument(
        '--lora',
        type=str,
        help='Download LoRA weights (provide HuggingFace ID, e.g., username/lora-name)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for models'
    )
    
    args = parser.parse_args()
    
    if not any([args.sdxl, args.svd, args.lora, args.all]):
        parser.print_help()
        logger.info("\nExample usage:")
        logger.info("  python setup_models.py --all")
        logger.info("  python setup_models.py --sdxl --svd")
        logger.info("  python setup_models.py --lora username/lora-name")
        return
    
    success = True
    
    if args.all or args.sdxl:
        success &= download_sdxl_model(cache_dir=args.cache_dir)
    
    if args.all or args.svd:
        success &= download_svd_model(cache_dir=args.cache_dir)
    
    if args.lora:
        success &= download_lora(args.lora)
    
    if success:
        logger.info("\n✓ All requested models downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update config.yaml with your LoRA path (if using LoRA)")
        logger.info("2. Run: python main.py your_image.jpg 'your prompt'")
    else:
        logger.error("\n✗ Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()

