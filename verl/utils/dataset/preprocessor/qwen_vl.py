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

"""
The Qwen-VL preprocessor used for the multi-modal models.
"""
from io import BytesIO
from typing import Optional, Union

from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER

__all__ = ["QwenVLPreprocessor"]


@PREPROCESSOR_REGISTER.register()
class QwenVLPreprocessor(BasicPreprocessor):
    """Preprocessor for Qwen-VL models (compatible with existing implementation)."""

    def __init__(self, processor, image_key="image", video_key="video", **kwargs):
        super().__init__(processor, image_key=image_key, video_key=video_key)

    def process_image(self, image: Union[dict, Image.Image]) -> Image.Image:
        """Process image using qwen_vl_utils."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if "bytes" in image:
            assert "image" not in image, "Cannot have both `bytes` and `image`"
            image["image"] = BytesIO(image["bytes"])

        return fetch_image(image)

    def process_video(
        self,
        video: dict,
        nframes: Optional[int] = None,
        fps: Optional[float] = None,
        fps_min_frames: Optional[int] = None,
        fps_max_frames: Optional[int] = None,
    ):
        """Process video using qwen_vl_utils."""
        assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

        # Shallow copy... since we might want to add some keys
        video = dict(video)

        contains_sampling_rules = "nframes" in video or "fps" in video
        if not contains_sampling_rules:
            if nframes is not None:
                video["nframes"] = nframes
            elif fps is not None:
                video["fps"] = fps
                if fps_min_frames is not None:
                    video["min_frames"] = fps_min_frames
                if fps_max_frames is not None:
                    video["max_frames"] = fps_max_frames

        return fetch_video(video)

    def process_audio(self, audio, **kwargs):
        """Qwen-VL does not support audio."""
        raise ValueError("Qwen-VL does not support audio")



