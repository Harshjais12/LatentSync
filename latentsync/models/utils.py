# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from codeformer import CodeFormer

def apply_super_resolution(subframe, method="GFPGAN"):
    if method == "GFPGAN":
        gfpgan = GFPGANer(model_path="checkpoints/GFPGAN.pth")
        _, restored_img, _ = gfpgan.enhance(subframe, has_aligned=False, only_center_face=False, paste_back=True)
        return restored_img
    elif method == "CodeFormer":
        codeformer = CodeFormer(model_path="checkpoints/CodeFormer.pth")
        restored_img = codeformer.enhance(subframe)
        return restored_img
    return subframe  
