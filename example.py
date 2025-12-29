#!/usr/bin/env python3
"""
Example script demonstrating how to use the AI Video Generation Pipeline
"""

from src.pipeline import VideoGenerationPipeline
from pathlib import Path

def main():
    # Initialize the pipeline
    print("Initializing pipeline...")
    pipeline = VideoGenerationPipeline(config_path="config.yaml")
    
    # Example: Generate video from an image
    input_image = "path/to/your/image.jpg"  # Update this path
    prompt = "a beautiful sunset over mountains, cinematic, high quality"
    negative_prompt = "blurry, low quality, distorted, artifacts"
    
    # Check if input image exists
    if not Path(input_image).exists():
        print(f"\n⚠️  Please update the 'input_image' path in this script.")
        print(f"   Current path: {input_image}")
        print("\nExample usage:")
        print("   python main.py your_image.jpg 'your prompt here'")
        return
    
    print(f"\nGenerating video from: {input_image}")
    print(f"Prompt: {prompt}\n")
    
    try:
        # Generate the video
        output_path = pipeline.generate(
            input_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_name="example_video",
            save_intermediate=True
        )
        
        print(f"\n✅ Success! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all models are downloaded")
        print("2. Check that your LoRA path is correct in config.yaml")
        print("3. Verify Wan I2V model path is set correctly")
        print("4. Ensure you have enough GPU memory")

if __name__ == "__main__":
    main()

