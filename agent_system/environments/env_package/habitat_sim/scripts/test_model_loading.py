#!/usr/bin/env python3
"""
Simple test script to verify model loading for both Qwen-VL and InternVL models.
This script tests the model loading functionality without running full evaluation.

Usage:
    python test_model_loading.py --model_path /path/to/model

Example:
    python test_model_loading.py --model_path /data1/tct_data/InternVL/InternVL3_5-2B-MPO
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modify as needed

import argparse
import torch
from PIL import Image
import numpy as np

# Import the loading function from eval_notebook
# Note: This assumes eval_notebook.py is in the same directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_system.environments.env_package.habitat_sim.eval.eval_grounding import load_model_and_processor, inference, load_internvl_image


def create_test_image(size=(480, 640)):
    """Create a simple test image for verification."""
    # Create a gradient image
    arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, size[1], dtype=np.uint8)  # Red gradient
    arr[:, :, 1] = np.linspace(0, 255, size[0], dtype=np.uint8).reshape(-1, 1)  # Green gradient
    arr[:, :, 2] = 128  # Constant blue
    
    # Add some text-like patterns
    arr[100:150, 200:400] = [255, 255, 255]  # White rectangle
    arr[200:250, 150:450] = [0, 0, 0]  # Black rectangle
    
    return Image.fromarray(arr)


def test_model_loading(model_path: str):
    """Test model and processor loading."""
    print("="*80)
    print("Testing Model Loading")
    print("="*80)
    print(f"Model path: {model_path}")
    print()
    
    try:
        # Test loading
        print("Loading model and processor...")
        model, processor, model_type = load_model_and_processor(model_path)
        print(f"✅ Successfully loaded model!")
        print(f"   Model type: {model_type}")
        print(f"   Processor type: {type(processor).__name__}")
        print(f"   Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
        print(f"   Model dtype: {next(model.parameters()).dtype if hasattr(model, 'parameters') else 'N/A'}")
        print()
        
        # Check model capabilities
        print("Model capabilities:")
        print(f"   Has 'chat' method: {hasattr(model, 'chat')}")
        print(f"   Has 'generate' method: {hasattr(model, 'generate')}")
        print(f"   Processor has 'apply_chat_template': {hasattr(processor, 'apply_chat_template')}")
        print()
        
        # Test image preprocessing
        print("Testing image preprocessing...")
        test_image = create_test_image()
        print(f"   Test image size: {test_image.size}")
        
        if "internvl" in model_type:
            print("   Testing InternVL image preprocessing...")
            pixel_values = load_internvl_image(test_image, model.config)
            print(f"   ✅ Pixel values shape: {pixel_values.shape}")
            print(f"   ✅ Pixel values dtype: {pixel_values.dtype}")
            print(f"   ✅ Number of image patches: {pixel_values.shape[0]}")
        else:
            print("   Qwen-VL uses processor for image preprocessing")
        print()
        
        # Test simple inference
        print("Testing inference...")
        test_prompt = "<image>\nDescribe this image briefly."
        
        try:
            output = inference(
                model, 
                processor, 
                test_image, 
                test_prompt, 
                model_type, 
                max_new_tokens=50
            )
            print(f"✅ Inference successful!")
            print(f"   Output type: {type(output)}")
            if isinstance(output, list) and len(output) > 0:
                print(f"   Output length: {len(output[0])} characters")
                print(f"   Output preview: {output[0][:100]}...")
            else:
                print(f"   Output: {output}")
        except Exception as e:
            print(f"⚠️  Inference test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("="*80)
        print("✅ All tests completed!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test model loading for Qwen-VL and InternVL models"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the model checkpoint directory'
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        return 1
    
    # Run tests
    success = test_model_loading(args.model_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())






