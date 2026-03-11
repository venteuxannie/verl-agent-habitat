#!/usr/bin/env python3
"""
Detect pure black regions (RGB == 0) in an image and mark them on a blank image.
Used to check for holes/gaps in scanned and reconstructed scenes.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def detect_black_regions(image_path, output_path=None):
    """
    Detect pure black regions (RGB == 0) in an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to output image, if None will be auto-generated
    
    Returns:
        mask: Binary mask, 1 for pure black regions (RGB == 0), 0 for others
        output_image: Blank image with pure black regions marked
    """
    # Read image
    print(f"Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        # Try reading with PIL
        img_pil = Image.open(image_path)
        img = np.array(img_pil)
        # PIL reads RGB, cv2 uses BGR, need to convert
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    print(f"Image shape: {img.shape}")
    
    # Convert to RGB (if image is in BGR format)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # cv2 reads BGR by default, convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()
    
    # Detect pure black regions: RGB channels all equal to 0
    # For RGB images, check if R, G, B values are all 0
    if len(img_rgb.shape) == 3:
        # Check if all channels equal 0
        black_mask = (img_rgb[:, :, 0] == 0) & \
                     (img_rgb[:, :, 1] == 0) & \
                     (img_rgb[:, :, 2] == 0)
    else:
        # Grayscale image
        black_mask = img_rgb == 0
    
    # Convert to uint8 format mask
    black_mask_uint8 = black_mask.astype(np.uint8) * 255
    
    # Statistics
    total_pixels = black_mask.size
    black_pixels = np.sum(black_mask)
    black_percentage = (black_pixels / total_pixels) * 100
    
    print(f"\nDetection results:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Pure black pixels (RGB == 0): {black_pixels}")
    print(f"  Pure black region percentage: {black_percentage:.2f}%")
    
    # Create blank image (white background)
    height, width = img_rgb.shape[:2]
    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Mark pure black regions on blank image (in red)
    blank_image[black_mask] = [255, 0, 0]  # Red color for pure black regions
    
    # Save results
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        dir_name = os.path.dirname(image_path)
        output_path = os.path.join(dir_name, f"{base_name}_black_regions.png")
    
    # Save marked image
    output_image_pil = Image.fromarray(blank_image)
    output_image_pil.save(output_path)
    print(f"\nResult saved to: {output_path}")
    
    # Also save mask image (black and white)
    mask_output_path = output_path.replace('.png', '_mask.png')
    mask_image = Image.fromarray(black_mask_uint8)
    mask_image.save(mask_output_path)
    print(f"Mask image saved to: {mask_output_path}")
    
    return black_mask_uint8, blank_image

def visualize_comparison(original_path, mask, output_path=None):
    """
    Visualize comparison between original image and detection results.
    """
    # Read original image
    original = Image.open(original_path)
    original_np = np.array(original)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Pure Black Regions Mask', fontsize=12)
    axes[1].axis('off')
    
    # Overlay mask on original image
    overlay = original_np.copy()
    if len(overlay.shape) == 3:
        overlay[mask > 0] = [255, 0, 0]  # Red color for marking
    else:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        overlay[mask > 0] = [255, 0, 0]
    
    axes[2].imshow(overlay)
    axes[2].set_title('Original + Pure Black Regions', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        dir_name = os.path.dirname(original_path)
        output_path = os.path.join(dir_name, f"{base_name}_black_regions_comparison.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect pure black regions (RGB == 0) in an image')
    parser.add_argument('--image', '-i', type=str, 
                       default='original_rgb.png',
                       help='Input image path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output image path (auto-generated if not specified)')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Do not generate comparison figure')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = args.image if os.path.isabs(args.image) else os.path.join(script_dir, args.image)
    
    # Detect pure black regions (RGB == 0)
    mask, blank_image = detect_black_regions(
        image_path, 
        output_path=args.output
    )
    
    # Generate comparison figure
    if not args.no_comparison:
        visualize_comparison(image_path, mask)

