"""FreeCD inference: semantic change detection on bi-temporal image pairs."""

import argparse
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datasets import get_dataset_meta
from freecd import FreeCD

DEFAULT_UPSAMPLER = './weight/dinov2_jbu_stack_million_aid_attention_crf_0_tv_0.0_ent_0.0_31500.ckpt'
DEFAULT_SEG_UPSAMPLER = './SegEarth/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'


def build_transform(size=448):
  return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Resize((size, size)),
  ])


def load_pair(t1_path, t2_path, transform, device, size=448):
  img1 = Image.open(t1_path).convert('RGB')
  img2 = Image.open(t2_path).convert('RGB')
  img1_rs = img1.resize((size, size))
  img2_rs = img2.resize((size, size))
  t1 = transform(img1).unsqueeze(0).to(device)
  t2 = transform(img2).unsqueeze(0).to(device)
  t1_bgr = cv2.cvtColor(np.array(img1_rs), cv2.COLOR_RGB2BGR)
  t2_bgr = cv2.cvtColor(np.array(img2_rs), cv2.COLOR_RGB2BGR)
  return torch.cat([t1, t2], dim=0), t1_bgr, t2_bgr


def apply_cd_mask(base_bgr, binary_mask):
  """Keep pixels where mask==255; set non-change regions to white."""
  base_bgr = np.ascontiguousarray(base_bgr)
  if base_bgr.shape[:2] != binary_mask.shape[:2]:
    binary_mask = cv2.resize(
      binary_mask,
      (base_bgr.shape[1], base_bgr.shape[0]),
      interpolation=cv2.INTER_NEAREST,
    )
  result = np.full_like(base_bgr, 255)
  change = binary_mask > 0
  result[change] = base_bgr[change]
  return result


def save_results(model, t1_sem, t2_sem, binary_mask, out_dir, name, t1_bgr, t2_bgr):
  out_dir = Path(out_dir)
  for sub in ('from', 'to', 'cd', 'from_color', 'to_color'):
    (out_dir / sub).mkdir(parents=True, exist_ok=True)

  mask = binary_mask.astype(np.uint8)
  t1_color = model.convert(t1_sem).astype('uint8')
  t2_color = model.convert(t2_sem).astype('uint8')

  cv2.imwrite(str(out_dir / 'cd' / name), mask)
  cv2.imwrite(str(out_dir / 'from' / name), apply_cd_mask(t1_bgr, mask))
  cv2.imwrite(str(out_dir / 'to' / name), apply_cd_mask(t2_bgr, mask))
  cv2.imwrite(str(out_dir / 'from_color' / name), apply_cd_mask(t1_color, mask))
  cv2.imwrite(str(out_dir / 'to_color' / name), apply_cd_mask(t2_color, mask))


def run_demo(args, model, transform, device):
  t1_path = args.t1 or 'image/T1.png'
  t2_path = args.t2 or 'image/T2.png'
  out_dir = args.output or 'output/demo'
  name = args.name or 'result.png'

  img, t1_bgr, t2_bgr = load_pair(t1_path, t2_path, transform, device, args.size)
  with torch.no_grad():
    t1_sem, t2_sem, binary_mask = model.predict(img)
  save_results(model, t1_sem, t2_sem, binary_mask, out_dir, name, t1_bgr, t2_bgr)
  print(f'Demo results saved to {out_dir}')


def run_batch(args, model, transform, device):
  t1_dir = Path(args.t1_dir)
  t2_dir = Path(args.t2_dir)
  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)

  names = sorted(os.listdir(t1_dir))
  with torch.no_grad():
    for i, name in enumerate(names):
      t1_path = t1_dir / name
      t2_path = t2_dir / name
      if not t2_path.exists():
        print(f'Skip {name}: missing in T2 directory')
        continue
      if (out_dir / 'cd' / name).exists():
        continue

      img, t1_bgr, t2_bgr = load_pair(t1_path, t2_path, transform, device, args.size)
      t1_sem, t2_sem, binary_mask = model.predict(img)
      save_results(model, t1_sem, t2_sem, binary_mask, out_dir, name, t1_bgr, t2_bgr)
      print(f'[{i + 1}/{len(names)}] {name}')
      gc.collect()
      torch.cuda.empty_cache()


def parse_args():
  parser = argparse.ArgumentParser(description='FreeCD inference')
  parser.add_argument('--dataset', default='SECOND', choices=list(__import__('datasets').DATASETS))
  parser.add_argument('--backbone', default='dinov2', help='BCD featurizer: dinov2, dinov3, CLIP, etc.')
  parser.add_argument('--upsampler', default='jbu_stack', help='BCD upsampler: jbu_stack, jbu_one, bilinear')
  parser.add_argument('--upsampler-ckpt', default=DEFAULT_UPSAMPLER, help='Path to BCD upsampler checkpoint')
  parser.add_argument('--size', type=int, default=448, help='Input resize (square)')
  parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
  parser.add_argument('--output', default=None, help='Output directory')

  parser.add_argument('--t1', default=None, help='T1 image path (demo mode)')
  parser.add_argument('--t2', default=None, help='T2 image path (demo mode)')
  parser.add_argument('--name', default='result.png', help='Output filename in demo mode')
  parser.add_argument('--t1-dir', default=None, help='T1 image directory (batch mode)')
  parser.add_argument('--t2-dir', default=None, help='T2 image directory (batch mode)')
  return parser.parse_args()


def main():
  args = parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  dataset_meta, cfg = get_dataset_meta(args.dataset)
  feature_up_cfg = dict(
    model_name=args.upsampler,
    model_path=args.upsampler_ckpt,
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = FreeCD(
    args.backbone,
    'SegEarth',
    name_path=cfg['name_path'],
    dataset_meta=dataset_meta,
    is_vis=False,
    is_ori=cfg['is_ori'],
    num_classes=cfg['num_classes'],
    feature_up_cfg=feature_up_cfg,
  ).to(device).eval()

  transform = build_transform(args.size)

  if args.t1_dir and args.t2_dir:
    run_batch(args, model, transform, device)
  else:
    run_demo(args, model, transform, device)


if __name__ == '__main__':
  main()
