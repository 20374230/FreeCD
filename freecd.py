from pyexpat import model
from matplotlib.scale import LogitScale
import torch
import torch.nn as nn
import sys

sys.path.append("..")



from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model

import sys
sys.path.insert(0, '../')

import gc
import os

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
from BCD import BCD
from SCD import SCD
@MODELS.register_module()
class FreeCD(nn.Module):
    def __init__(self,
                 bcd_name,
                 scd_name,
                 name_path='SECOND',
                 dataset_meta=dict( 
        classes=('unchanged', 'low vegetation','nvg surface','tree','water','building','playground'),
        palette=[[255, 255, 255],[0,128,0],[128,128,128],[0,255,0],[0,0,255],[128,0,0],[255,0,0]]),
                 is_vis=True,
                 is_ori=False,
                 device=torch.device('cuda'),
                 num_classes=6,
                 feature_up_cfg=dict(
                    #  model_name='jbu_stack2',
                     #model_name='jbu_one',
                     model_name='jbu_stack',
                    #  model_path='./SimFeatUp-main/work_dirs/simfeatup_million_aid/checkpoints/stack/dinov2_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0/work_dirs/simfeatup_million_aid/checkpoints/stack/dinov2_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0_900.ckpt')):
                   # model_path='./SimFeatUp-main/work_dirs/my/checkpoints/stack_0/dinov2_jbu_stack_million_aid_attention_crf_0.001_tv_0.0_ent_0.0/work_dirs/my/checkpoints/stack_0/dinov2_jbu_stack_million_aid_attention_crf_0.001_tv_0.0_ent_0.0_30000.ckpt')):
        # model_path='./SimFeatUp-main/work_dirs/my2/checkpoints/stack_0/dinov2_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0/work_dirs/my2/checkpoints/stack_0/dinov2_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_21900.ckpt')):
        # model_path='./work_dirs/v3/checkpoints/stack_0/dinov3_jbu_stack_million_aid_attention_crf_0.001_tv_0.0_ent_0.0/work_dirs/v3/checkpoints/stack_0/dinov3_jbu_stack_million_aid_attention_crf_0.001_tv_0.0_ent_0.0_31500.ckpt')):
                # model_path='./work_dirs/v3-sim/checkpoints/stack_0/dinov3_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0/work_dirs/v3-sim/checkpoints/stack_0/dinov3_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0_31500.ckpt')):
# model_path='./work_dirs/v3-sim-o/checkpoints/stack_0/dinov3_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0/work_dirs/v3-sim-o/checkpoints/stack_0/dinov3_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_4500.ckpt')):
        model_path='./weight/dinov2_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0_31500.ckpt')):
        # model_path='/hdd1/zys/wnmd/CD3/work_dirs/v3-FNO-new2/checkpoints/stack_0/dinov3_jbu_stack2_million_aid_attention_crf_0_tv_0.0_ent_0.0/work_dirs/v3-FNO-new2/checkpoints/stack_0/dinov3_jbu_stack2_million_aid_attention_crf_0_tv_0.0_ent_0.0_19400.ckpt')):#stack2

        super(FreeCD,self).__init__()
        self.bcd=BCD(bcd_name, feature_up_cfg=feature_up_cfg, is_vis=is_vis)
        self.scd=SCD(scd_name,name_path=name_path)
        self.dataset_meta=dataset_meta
        self.is_vis=is_vis
        self.is_ori=is_ori
        self.num_classes=num_classes
    def my_argmax(self,i_seg_logits,dim=0):
        _,l=torch.max(i_seg_logits[:,(dim+1):self.num_classes+1, :, :], dim=1,keepdim=True)
        return l+1
    def convert(self,img):
        palette=self.dataset_meta['palette']
        image = np.expand_dims(img, axis=2)
        image = np.concatenate((image, image, image), axis=-1)  #-1则是最后一个维度 
        for id,v in enumerate(palette):
            image[np.all(image==[id,id,id], axis=-1)]=[v[2],v[1],v[0]]
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
    def postprocess(self,logits,logits_scd):
        if not self.is_ori:
            T1_semantic=self.my_argmax(logits_scd[:1,:,:,:]).squeeze(0).squeeze(0).cpu().numpy()
            T2_semantic=self.my_argmax(logits_scd[1:,:,:,:]).squeeze(0).squeeze(0).cpu().numpy()
        else:
            T1_semantic=torch.argmax(logits_scd[:1,:,:,:],dim=1,keepdim=True).squeeze(0).squeeze(0).cpu().numpy()
            T2_semantic=torch.argmax(logits_scd[1:,:,:,:],dim=1,keepdim=True).squeeze(0).squeeze(0).cpu().numpy()
        p=logits.squeeze(0).squeeze(0)
        cos_similarity_flat = p.reshape(-1).cpu().detach().numpy()
        threshold = threshold_otsu(cos_similarity_flat)
        binary_mask = np.where(p.cpu().detach().numpy() > threshold, 255, 0)

        # ȷ����������Ϊuint8��PNGͼ��Ҫ��
        binary_mask = binary_mask.astype(np.uint8)
        return T1_semantic,T2_semantic,binary_mask
    def predict(self,img):
        batch_img_metas = [
                        dict(
                            ori_shape=img.shape[2:],
                            img_shape=img.shape[2:],
                            pad_shape=img.shape[2:],
                            padding_size=[0, 0, 0, 0])
                    ] * img.shape[0]
        logits_bcd=self.bcd.forward_slide(img,batch_img_metas)
        logits_scd=torch.softmax(self.scd(img),dim=1)
        logits=logits_bcd*(1-(logits_scd[1:,:1,:,:])*(logits_scd[:1,:1,:,:]))
        # logits_scd=self.my_argmax(logits_scd)
        return self.postprocess(logits,logits_scd)
    def __call__(self, img):
        batch_img_metas = [
                        dict(
                            ori_shape=img.shape[2:],
                            img_shape=img.shape[2:],
                            pad_shape=img.shape[2:],
                            padding_size=[0, 0, 0, 0])
                    ] * img.shape[0]
        logits_bcd=self.bcd.forward_slide(img,batch_img_metas)
        logits_scd=torch.softmax(self.scd(img),dim=1)
        logits=logits_bcd*(1-logits_scd[1:,:1,:,:])+logits_bcd*(1-logits_scd[:1,:1,:,:])
        # logits_scd=self.my_argmax(logits_scd)
        return logits,logits_scd



# img1=Image.open("image/T1.png").convert("RGB")
# img2=Image.open("image/T2.png").convert("RGB")
# transform= transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     transforms.Resize((448, 448))
# ])
# img1_tensor=transform(img1).unsqueeze(0).cuda()
# img2_tensor=transform(img2).unsqueeze(0).cuda()
# img=torch.cat([img1_tensor,img2_tensor],dim=0)
# model=FreeCD("dinov2","SegEarth")
# with torch.no_grad():
#     batch_img_metas = [
#                             dict(
#                                 ori_shape=img.shape[2:],
#                                 img_shape=img.shape[2:],
#                                 pad_shape=img.shape[2:],
#                                 padding_size=[0, 0, 0, 0])
#                         ] * img.shape[0]
#     p,p2=model(img)
#     p=p.squeeze(0).squeeze(0)
#     cos_similarity_flat = p.reshape(-1).cpu().detach().numpy()
#     threshold = threshold_otsu(cos_similarity_flat)
#     binary_mask = np.where(p.cpu().detach().numpy() > threshold, 255, 0)

#     # ȷ����������Ϊuint8��PNGͼ��Ҫ��
#     binary_mask = binary_mask.astype(np.uint8)

#     # ����PILͼ�񲢱���
#     mask_image = Image.fromarray(binary_mask)
#     mask_image.save("out/change_mask.png")