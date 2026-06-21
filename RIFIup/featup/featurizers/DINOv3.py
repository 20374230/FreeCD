import math
import warnings
from functools import partial
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
import torch
import torch.nn as nn

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
import torch
from transformers import AutoModel
from transformers.image_utils import load_image

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov3.layers import Mlp, PatchEmbed


logger = logging.getLogger("dinov3")


class DINOv3Featurizer(nn.Module):

    def __init__(self, arch, patch_size, feat_type):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type
        
        self.n_feats = 128
        self.model =  torch.hub.load('dinov3-main', 'dinov3_vitl16', source='local', weights='/hdd1/zys/Dinov3_LLM/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

    def get_cls_token(self, img):
        return self.model.forward(img)

    def forward(self, img, n=1, include_cls=False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size
        return self.model.forward_features(img)["x_norm_patchtokens"].reshape(-1, h, w, 1024).permute(0, 3, 1, 2)
