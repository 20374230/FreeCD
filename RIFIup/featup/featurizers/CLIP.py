import clip
import torch
from torch import nn
import os
from open_clip import tokenizer, create_model

class CLIPFeaturizer(nn.Module):

    def __init__(self,name):
        super().__init__()
        if name=='CLIP':
            self.model = create_model('ViT-B/16', pretrained='openai').visual
            self.model.output_tokens=True
            self.patch_size=[16,16]
            self.dim=512
        elif name=='RemoteCLIP':
        
            self.model = create_model('ViT-L-14', pretrained='/hdd1/zys/zys/.cache/torch/hub/checkpoints/RemoteCLIP-ViT-L-14.pt').visual
            self.model.output_tokens=True
            self.patch_size=[14,14]
            self.dim=768
        elif name=='RS5M':
            self.model = create_model('ViT-L-14', pretrained='/hdd1/zys/zys/.cache/torch/hub/checkpoints/RS5M_ViT-L-14.pt').visual
            self.model.output_tokens=True
            self.patch_size=[14,14]
            self.dim=768
    def get_cls_token(self, img):
        return self.model.encode_image(img).to(torch.float32)

    def forward(self, img):
        features=self.model(img)
        feature_w, feature_h = img.shape[-2] // self.patch_size[0], img.shape[-1] // self.patch_size[1]
        features = features.permute(0, 2, 1).view(2, self.dim, feature_w, feature_h)
        return features


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from shared import norm, crop_to_divisor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("../samples/lex1.jpg")
    load_size = 224  # * 3
    transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        # T.CenterCrop(load_size),
        T.ToTensor(),
        lambda x: crop_to_divisor(x, 16),
        norm])

    model = CLIPFeaturizer().cuda()

    results = model(transform(image).cuda().unsqueeze(0))

    print(clip.available_models())
