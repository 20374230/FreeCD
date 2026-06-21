import gc
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
import torch.nn.functional as F
from .datasets.JitteredImage import apply_jitter, sample_transform
from .datasets.util import get_dataset, SingleImageDataset
from .downsamplers import SimpleDownsampler, AttentionDownsampler
from .featurizers.util import get_featurizer
from .layers import ChannelNorm
from .losses import TVLoss, SampledCRFLoss, entropy
from .upsamplers import get_upsampler, LayerNorm2d
from .util import pca, RollingAvg, unnorm, norm, prep_image
from .FNO import ResolutionInvariantFNO
torch.multiprocessing.set_sharing_strategy('file_system')


class ScaleNet(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = torch.nn.Conv2d(dim, 1, 1)
        with torch.no_grad():
            self.net.weight.copy_(self.net.weight * .1)
            self.net.bias.copy_(self.net.bias * .1)

    def forward(self, x):
        return torch.exp(self.net(x) + .1).clamp_min(.0001)


class FNO(pl.LightningModule):
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 predicted_uncertainty,
                 crf_weight,
                 filter_ent_weight,
                 tv_weight,
                 rec_img_weight,
                 upsampler,
                 downsampler,
                 chkpt_dir,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.predicted_uncertainty = predicted_uncertainty
        self.crf_weight = crf_weight
        self.filter_ent_weight = filter_ent_weight
        self.tv_weight = tv_weight
        self.rec_img_weight = rec_img_weight
        self.chkpt_dir = chkpt_dir

        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        for p in self.model.parameters():
            p.requires_grad = False
        # self.model = torch.nn.Sequential(self.model, ChannelNorm(self.dim))
        #self.upsampler = get_upsampler(upsampler, self.dim)
        self.FNO=ResolutionInvariantFNO(
        input_channels=self.dim,
        output_channels=3,
        width=64,
        modes=8  # 对于16x16输入，模态数不能太大
    )
        # if downsampler == 'simple':
        #     self.downsampler = SimpleDownsampler(self.kernel_size, self.final_size)
        # elif downsampler == 'attention':
        #     self.downsampler = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
        # else:
        #     raise ValueError(f"Unknown downsampler {downsampler}")

        # if self.predicted_uncertainty:
        #     self.scale_net = ScaleNet(self.dim)

        self.avg = RollingAvg(20)

        # self.crf = SampledCRFLoss(
        #     alpha=.1,
        #     beta=.15,
        #     gamma=.005,
        #     w1=10.0,
        #     w2=3.0,
        #     shift=0.00,
        #     n_samples=1000)
        # self.tv = TVLoss()
        self.mse_loss=torch.nn.MSELoss()

        # if self.rec_img_weight!=0:
        #     self.projection_img = torch.nn.Sequential(
        #         torch.nn.Conv2d(self.dim, self.dim, 1),
        #         LayerNorm2d(self.dim),
        #         torch.nn.GELU(),
        #         torch.nn.Conv2d(self.dim, 3, 1),
        #         torch.nn.Tanh()
        #     )


    def forward(self, x):
        return self.FNO(self.model(x))

    def project(self, feats, proj):
        if proj is None:
            return feats
        else:
            return torch.einsum("bchw,bcd->bdhw", feats, proj)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if type(batch) == dict:
                img = batch['img']
            else:
                img, _ = batch
            feats = self.model(img)

        rec_img=self.FNO(feats)
        targets=F.interpolate(img, size=(int(224/self.patch_size),int(224/self.patch_size)), mode='bicubic' )      
        loss=self.mse_loss(rec_img, targets)
        full_total_loss=0    
        full_total_loss += loss.item()
        self.manual_backward(loss)

        #self.avg.add('loss/rec_img', full_rec_img_loss)
        self.avg.add("loss/total", full_total_loss)

        if self.global_step % 10000 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + '/' + self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        self.avg.logall(self.log)
        if self.global_step < 10:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.0001)

        opt.step()

        return None

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def on_save_checkpoint(self, checkpoint):
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if 'FNO' in key:
                new_state_dict[key] = value
        checkpoint['state_dict'] = new_state_dict

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.trainer.is_global_zero and batch_idx == 0:

                if type(batch) == dict:
                    img = batch['img']
                else:
                    img, _ = batch
                lr_feats = self.model(img)
                writer = self.logger.experiment
                writer.flush()

    def configure_optimizers(self):
        all_params = []
        all_params.extend(list(self.FNO.parameters()))


        return torch.optim.NAdam(all_params, lr=self.lr)


@hydra.main(config_path="configs", config_name="FNO_train.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    load_size = 224

    if cfg.model_type == "dinov2":
        final_size = 16
        kernel_size = 14
    elif cfg.model_type == "maskclip_vit-b32":
        final_size = 7
        kernel_size = 32
    else:
        final_size = 14
        kernel_size = 16

    name = (f"{cfg.model_type}_{cfg.upsampler_type}_"
            f"{cfg.dataset}_{cfg.downsampler_type}_"
            f"crf_{cfg.crf_weight}_tv_{cfg.tv_weight}"
            f"_ent_{cfg.filter_ent_weight}")

    log_dir = join(cfg.output_root, f"logs/stack_0/{name}")
    chkpt_dir = join(cfg.output_root, f"checkpoints/stack_0/{name}.ckpt")
    os.makedirs(log_dir, exist_ok=True)

    model = FNO(
        model_type=cfg.model_type,
        activation_type=cfg.activation_type,
        n_jitters=cfg.n_jitters,
        max_pad=cfg.max_pad,
        max_zoom=cfg.max_zoom,
        kernel_size=kernel_size,
        final_size=final_size,
        lr=cfg.lr,
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        crf_weight=cfg.crf_weight,
        filter_ent_weight=cfg.filter_ent_weight,
        tv_weight=cfg.tv_weight,
        rec_img_weight=cfg.rec_img_weight,
        upsampler=cfg.upsampler_type,
        downsampler=cfg.downsampler_type,
        chkpt_dir=chkpt_dir
    )

    transform = T.Compose([
        # T.Resize(load_size, InterpolationMode.BILINEAR),
        # T.CenterCrop(load_size),
        T.RandomCrop(load_size, pad_if_needed=True, padding_mode='reflect'),
        # T.RandomResizedCrop(load_size, ratio=(0.8, 1.2)),
        T.ToTensor(),
        norm])

    dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        transform=transform)

    loader = DataLoader(
        dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    val_loader = DataLoader(
        SingleImageDataset(0, dataset, 1), 1, shuffle=False, num_workers=cfg.num_workers)

    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]
    # callbacks = [EarlyStopping(monitor="loss/total", mode="min")]

    pl_major = int(pl.__version__.split('.')[0])
    if pl_major >= 2:
        trainer = Trainer(
            accelerator='gpu',
            strategy='ddp' if cfg.num_gpus > 1 else 'auto',
            devices=cfg.num_gpus,
            max_epochs=cfg.epochs,
            logger=tb_logger,
            val_check_interval=100,
            log_every_n_steps=10,
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,
        )
    else:
        trainer = Trainer(
            gpus=cfg.num_gpus,
            max_epochs=cfg.epochs,
            logger=tb_logger,
            val_check_interval=100,
            log_every_n_steps=10,
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,
        )

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    trainer.fit(model, loader, val_loader)
    trainer.save_checkpoint(chkpt_dir)


if __name__ == "__main__":
    my_app()
