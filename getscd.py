from SegEarth.segearth_segmentor import SegEarthSegmentation

CLS_FILES = {
    'SECOND': './cls_SECOND.txt',
    'HIUCD': './cls_HIUCD.txt',
    'WHU': './cls_WHU.txt',
    'HRSCD': './cls_HRSCD.txt',
    'JLSCD': './cls_JLSCD.txt',
    'LEVIRSCD': './cls_LEVIRSCD.txt',
}


def get_segmentor(segmentor, name_path='SECOND'):
  if segmentor != 'SegEarth':
    raise ValueError(f"Only SegEarth segmentor is supported in this release, got '{segmentor}'")

  if name_path not in CLS_FILES:
    raise ValueError(f"Unknown dataset '{name_path}'. Choose from: {list(CLS_FILES)}")

  return SegEarthSegmentation(
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='SegEarth',
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
      model_name='jbu_one',
      model_path='./SegEarth/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt',
    ),
    cls_token_lambda=-0.3,
    name_path=CLS_FILES[name_path],
    prob_thd=0.1,
  )
