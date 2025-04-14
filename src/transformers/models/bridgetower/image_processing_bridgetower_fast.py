# coding=utf-8
# Copyright 2025 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for BridgeTower."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import add_start_docstrings


@add_start_docstrings(
    "Constructs a fast BridgeTower image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class BridgeTowerImageProcessorFast(BaseImageProcessorFast):
    # 기본 값: slow processor와 동일하게 설정하도록 조정합니다.
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 288}
    size_divisor = 32  # slow processor와 일치하도록 size_divisor 추가
    default_to_square = False
    crop_size = {"height": 288, "width": 384}  # slow processor의 center crop 결과에 맞춰 수정
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_pad = True  # 필요 시 패딩 적용
    do_convert_rgb = None


__all__ = ["BridgeTowerImageProcessorFast"]
