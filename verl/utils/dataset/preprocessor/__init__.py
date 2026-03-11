# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from transformers import AutoProcessor

from .base_processor import BasicPreprocessor
from .internvl import InternVLPreprocessor
from .qwen_vl import QwenVLPreprocessor
from .registry import PREPROCESSOR_REGISTER

__all__ = ["BasicPreprocessor", "InternVLPreprocessor", "QwenVLPreprocessor", "map_processor_to_preprocessor", "PREPROCESSOR_REGISTER"]


def map_processor_to_preprocessor(processor: AutoProcessor):
    """Map the processor to the Preprocessor class.
    
    Args:
        processor (AutoProcessor): The processor.
        
    Returns:
        class: The preprocessor class
    """
    processor_name = processor.__class__.__name__
    
    if not processor_name.lower().endswith("processor"):
        raise ValueError(f"Source object '{processor_name}' is not a 'Processor'.")
    
    # Special handling for Qwen2/Qwen2.5 VL processors
    if re.match(r"Qwen2.*?VLProcessor", processor_name):
        print("Qwen-VL2 Series will use the QwenVLPreprocessor")
        dest_name = "QwenVLPreprocessor".lower()
    else:
        # Convert XxxProcessor to XxxPreprocessor
        dest_name = processor_name.lower().replace("processor", "preprocessor")
    
    try:
        dest_class = PREPROCESSOR_REGISTER.get(dest_name)
    except KeyError:
        # Fallback to BasicPreprocessor if specific preprocessor not found
        print(f"Warning: No specific preprocessor found for {processor_name}, using BasicPreprocessor")
        dest_class = BasicPreprocessor
    
    return dest_class



