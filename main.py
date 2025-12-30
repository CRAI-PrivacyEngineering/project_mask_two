#!/usr/bin/env python3
"""
Main entry point for AI Video Generation Tool
"""

import argparse
import sys
import re
import tempfile
import subprocess
from pathlib import Path
from PIL import Image
import logging

from src.pipeline import VideoGenerationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_youtube_url(url):
    """Check if the input is a YouTube URL"""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


def download_youtube_frame(url, output_path=None, frame_time="00:00:01"):
    """
    Download a YouTube video and extract a frame
    
    Args:
        url: YouTube URL
        output_path: Path to save the extracted frame (optional)
        frame_time: Time to extract frame from (HH:MM:SS format)
    
    Returns:
        Path to the extracted frame image
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.jpg', prefix='youtube_frame_')
    
    # Convert frame_time to seconds for opencv
    time_parts = frame_time.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(float, time_parts)
        frame_seconds = int(hours * 3600 + minutes * 60 + seconds)
    else:
        frame_seconds = int(float(frame_time))
    
    # Try yt-dlp first (more modern), fall back to youtube-dl
    for tool in ['yt-dlp', 'youtube-dl']:
        try:
            # Check if tool is available
            subprocess.run([tool, '--version'], 
                         capture_output=True, check=True, timeout=5)
            
            logger.info(f"Downloading YouTube video using {tool}...")
            
            # Download video to temporary file
            temp_video = tempfile.mktemp(suffix='.mp4', prefix='youtube_video_')
            
            result = subprocess.run(
                [tool, '-f', 'best[height<=720]/best', '-o', temp_video, url],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and Path(temp_video).exists():
                # Use opencv to extract frame
                try:
                    import cv2
                    logger.info(f"Extracting frame at {frame_time} ({frame_seconds} seconds)...")
                    
                    cap = cv2.VideoCapture(temp_video)
                    cap.set(cv2.CAP_PROP_POS_MSEC, frame_seconds * 1000)
                    
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(output_path, frame)
                        cap.release()
                        # Clean up temp video
                        try:
                            Path(temp_video).unlink()
                        except:
                            pass
                        logger.info(f"Frame extracted to: {output_path}")
                        return output_path
                    else:
                        cap.release()
                        raise RuntimeError(f"Could not read frame at {frame_time}")
                except ImportError:
                    logger.warning("opencv-python not available, trying ffmpeg...")
                    # Fall through to ffmpeg method
                except Exception as e:
                    logger.warning(f"OpenCV extraction failed: {e}, trying ffmpeg...")
                    # Fall through to ffmpeg method
                
                # Try ffmpeg if opencv failed
                try:
                    subprocess.run(
                        ['ffmpeg', '-y', '-ss', frame_time, '-i', temp_video,
                         '-vframes', '1', '-q:v', '2', output_path],
                        capture_output=True,
                        check=True,
                        timeout=60
                    )
                    # Clean up temp video
                    try:
                        Path(temp_video).unlink()
                    except:
                        pass
                    logger.info(f"Frame extracted to: {output_path}")
                    return output_path
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    # Clean up temp video
                    try:
                        Path(temp_video).unlink()
                    except:
                        pass
                    raise RuntimeError(f"Failed to extract frame: {e}")
                    
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"{tool} not available or failed: {e}")
            continue
    
    raise RuntimeError(
        f"Failed to download/extract frame from YouTube URL. "
        f"Please ensure yt-dlp is installed: pip install yt-dlp\n"
        f"Or extract a frame manually and provide the image path."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic AI videos from images using SDXL LoRA, img2img, upscaling, and I2V (SVD/Wan2.2)"
    )
    
    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input image file or YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)'
    )
    
    parser.add_argument(
        '--frame-time',
        type=str,
        default='00:00:01',
        help='Time to extract frame from YouTube video (HH:MM:SS format, default: 00:00:01)'
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
    
    # Handle YouTube URLs
    input_path = args.input_image
    temp_frame_path = None
    
    if is_youtube_url(args.input_image):
        logger.info(f"Detected YouTube URL: {args.input_image}")
        try:
            temp_frame_path = download_youtube_frame(
                args.input_image,
                frame_time=args.frame_time
            )
            input_path = temp_frame_path
            logger.info(f"Using extracted frame: {input_path}")
        except Exception as e:
            logger.error(f"Failed to process YouTube URL: {e}")
            logger.error(
                "\nTo use YouTube URLs, please install yt-dlp:\n"
                "  pip install yt-dlp\n"
                "\nOr extract a frame manually:\n"
                "  yt-dlp -f 'best[height<=720]' -g 'YOUTUBE_URL' | xargs -I {} ffmpeg -ss 00:00:01 -i {} -vframes 1 frame.jpg"
            )
            sys.exit(1)
    else:
        # Validate input image file
        if not Path(args.input_image).exists():
            logger.error(f"Input image not found: {args.input_image}")
            sys.exit(1)
        input_path = args.input_image
    
    # Validate config file
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = VideoGenerationPipeline(config_path=args.config)
        
        # Generate video
        output_path = pipeline.generate(
            input_image=input_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_name=args.output_name,
            save_intermediate=not args.no_save_intermediate
        )
        
        logger.info(f"\n✓ Success! Video generated: {output_path}")
        
        # Clean up temporary frame if we downloaded from YouTube
        if temp_frame_path and Path(temp_frame_path).exists():
            try:
                Path(temp_frame_path).unlink()
                logger.debug(f"Cleaned up temporary frame: {temp_frame_path}")
            except Exception as e:
                logger.debug(f"Could not clean up temporary file: {e}")
        
    except Exception as e:
        logger.error(f"Error during video generation: {e}", exc_info=True)
        # Clean up on error
        if temp_frame_path and Path(temp_frame_path).exists():
            try:
                Path(temp_frame_path).unlink()
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()

