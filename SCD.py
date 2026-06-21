import gc
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pyexpat import model
from matplotlib.scale import LogitScale
import torch
import torch.nn as nn
import sys
import cv2
sys.path.append("..")



from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model

import sys
sys.path.insert(0, '../')

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from os.path import join
from torchvision import transforms
from RIFIup.featup.datasets.JitteredImage import apply_jitter, sample_transform
from RIFIup.featup.datasets.util import get_dataset, SingleImageDataset
from RIFIup.featup.downsamplers import SimpleDownsampler, AttentionDownsampler
from RIFIup.featup.featurizers.util import get_featurizer
from RIFIup.featup.layers import ChannelNorm
from RIFIup.featup.losses import TVLoss, SampledCRFLoss, entropy
from RIFIup.featup.upsamplers import get_upsampler, LayerNorm2d
from RIFIup.featup.util import pca, RollingAvg, unnorm, norm, prep_image
from RIFIup.featup.train_upsampler import RIFIup
from RIFIup.featup.upsamplers import get_upsampler, LayerNorm2d
from RIFIup.featup.featurizers.util import get_featurizer
from PIL import Image
import math
import torch
import numpy as np
import torch.nn.functional as F
from skimage.filters.thresholding import threshold_otsu
from getscd import get_segmentor
# import RSKT_Seg
# import GSNet
@MODELS.register_module()
class SCD():
    def __init__(self,
                 featureizer,
                 name_path='SECOND',
                 dataset_meta=dict( 
        classes=('unchanged', 'low vegetation','nvg surface','tree','water','building','playground'),
        palette=[[255, 255, 255],[0,128,0],[128,128,128],[0,255,0],[0,0,255],[128,0,0],[255,0,0]]),
                 device=torch.device('cuda')):
        self.featureizer=get_segmentor(featureizer,name_path)
        self.dataset_meta=dataset_meta
    def convert(self,img):
        palette=self.dataset_meta['palette']
        image = np.expand_dims(img, axis=2)
        image = np.concatenate((image, image, image), axis=-1)  #-1则是最后一个维度 
        for id,v in enumerate(palette):
            image[np.all(image==[id,id,id], axis=-1)]=[v[2],v[1],v[0]]
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
    def __call__(self, img):
      logits=self.featureizer(img)
      return logits



# img1=Image.open("image/T1.png").convert("RGB")
# img2=Image.open("image/T2.png").convert("RGB")
# transform= transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     transforms.Resize((224, 224))
# ])
# img1_tensor=transform(img1).unsqueeze(0).cuda()
# img2_tensor=transform(img2).unsqueeze(0).cuda()
# img=torch.cat([img1_tensor,img2_tensor],dim=0)
# model=SCD("SED")
# batch_img_metas = [
#                         dict(
#                             ori_shape=img.shape[2:],
#                             img_shape=img.shape[2:],
#                             pad_shape=img.shape[2:],
#                             padding_size=[0, 0, 0, 0])
#                     ] * img.shape[0]
# p=torch.argmax(model(img)[:,(0+1):6+1, :, :],dim=1)[0].cpu().numpy()
# p=model.convert(p)
# # ����PILͼ�񲢱���
# cv2.imwrite("out/p1.png",p)
# # mask_image = Image.fromarray(np.uint8(p))
# # mask_image.save("out/p1.png")