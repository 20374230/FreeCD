from pyexpat import model
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

@MODELS.register_module()
class BCD(BaseSegmentor):
    def __init__(self,
                 featureizer,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=None,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 is_vis=True,
                 feature_up_cfg=dict(

                     model_name='jbu_stack',
                     model_path='./weight/dinov2_jbu_stack.ckpt')):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.featurizer,self.patch_size,self.dim=get_featurizer(featureizer)
        
        self.featurizer.eval().to(device)
       
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx
        self.is_vis=is_vis


        
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if feature_up:
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.dim).cuda()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            if feature_up_cfg['model_name']!='bilinear':
                self.upsampler.load_state_dict(weights_dict, strict=True)

    def forward_feature(self, img, logit_size=None):
        img_tensor=self.featurizer(img)
        if self.feature_up:
            up_feat=self.upsampler(img_tensor,img)
        #vis_feat
        if self.is_vis:
            feat_t1=img_tensor[0].unsqueeze(0)
            feat_t2=img_tensor[1].unsqueeze(0)
            up_feat_t1=up_feat[0].unsqueeze(0)
            up_feat_t2=up_feat[1].unsqueeze(0)
            [t1_pca_low], fit_pca = pca([feat_t1[0].unsqueeze(0)])
            [t2_pca_low], _ = pca([feat_t2[0].unsqueeze(0)], fit_pca=fit_pca)
            [t1_pca_high], _ = pca([up_feat_t1[0].unsqueeze(0)], fit_pca=fit_pca)
            [t2_pca_high], _ = pca([up_feat_t2[0].unsqueeze(0)], fit_pca=fit_pca)
            b=transforms.ToPILImage()(t1_pca_low.squeeze(0))
            b.save("out/T1_low.jpg")
            b=transforms.ToPILImage()(t2_pca_low.squeeze(0))
            b.save("out/T2_low.jpg")
            b=transforms.ToPILImage()(t1_pca_high.squeeze(0))
            b.save("out/T1_high.jpg")
            b=transforms.ToPILImage()(t2_pca_high.squeeze(0))
            b.save("out/T2_high.jpg")
        #------
        logits = torch.sigmoid(-F.cosine_similarity(up_feat[0], up_feat[1], dim=0)).unsqueeze(0).unsqueeze(0)
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        batch_size=1
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
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
# model=BCD("dinov3")
# with torch.no_grad():
#     batch_img_metas = [
#                             dict(
#                                 ori_shape=img.shape[2:],
#                                 img_shape=img.shape[2:],
#                                 pad_shape=img.shape[2:],
#                                 padding_size=[0, 0, 0, 0])
#                         ] * img.shape[0]
#     p=model.forward_slide(img,batch_img_metas).squeeze(0).squeeze(0)
#     cos_similarity_flat = p.reshape(-1).cpu().detach().numpy()
#     threshold = threshold_otsu(cos_similarity_flat)
#     binary_mask = np.where(p.cpu().detach().numpy() > threshold, 255, 0)

#     # ȷ����������Ϊuint8��PNGͼ��Ҫ��
#     binary_mask = binary_mask.astype(np.uint8)

#     # ����PILͼ�񲢱���
#     mask_image = Image.fromarray(binary_mask)
#     mask_image.save("out/change_mask.png")