<p align="center">
<h1 align="center">Free-CD: Probabilistically Decoupled Training-Free Open-Vocabulary Change Detection with Resolution-Invariant Feature Inversion</h1>
</p>

<p align="center">
    <strong>Yongshuo Zhu, Lu Li, Keyan Chen, Zhenwei Shi, Fugen Zhou</strong>
</p>

<p align="center">
    Image Processing Center, School of Astronautics, Beihang University, Beijing 100191, China<br>
    State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, Beijing 100191, China
</p>

<p align="center">
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
</p>

---

## Abstract

Semantic Change Detection (SCD) transforms bi-temporal remote sensing data into actionable insights for urban planning, disaster response, and environmental conservation. While supervised learning achieves pixel-level precision, it remains restricted by fixed vocabularies. The dynamic nature of the physical world demands Open-Vocabulary Change Detection (OVCD).

However, OVCD faces a fundamental granularity gap: foundation models (e.g., CLIP, DINO) prioritize high-level semantic abstraction while change detection requires pixel-level spatial fidelity. Traditional OVCD methods rely on instance extraction models, introducing spatial semantic ambiguities and leading to over-segmentation or under-segmentation in remote sensing scenarios.

To address these challenges, we propose **Free-CD**, a unified SOTA framework. This release includes three main components: (1) data preprocessing and model preparation (datasets.py, getscd.py), (2) model core with BCD and SCD fusion (BCD.py, SCD.py, freecd.py), and (3) training and inference pipelines (inference.py). Extensive experiments validate our approach across multiple remote sensing datasets, establishing new state-of-the-art performance benchmarks in accuracy and generalization.

---

## Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.0
- CUDA 11.8+ (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/20374230/FreeCD.git
cd FreeCD

# Install dependencies
pip install -r requirements.txt
```

**Note:** This README covers the full pipeline from data to results. Backbone weights (DINOv2, DINOv3, CLIP, etc.) are downloaded automatically on first run.

---

## Methodology

### BCD: Binary Change Detection
```python
# core/BCD.py
class BCD:
    def forward_feature(self, img, logit_size=None):
        # Extract multi-scale features with DINO-series backbones
        img_tensor = self.featurizer(img)
        
        # Upsample with RIFI-Up if enabled
        if self.feature_up:
            up_feat = self.upsampler(img_tensor, img)
        
        # Compute change probability via cosine similarity
        logits = torch.sigmoid(-F.cosine_similarity(
            up_feat[0], up_feat[1], dim=0
        )).unsqueeze(0).unsqueeze(0)
        
        return logits
```

### SCD: Semantic Change Detection
```python
# core/SCD.py
class SCD:
    def __call__(self, img):
        # Open-vocabulary segmentation
        logits = self.featureizer(img)
        return logits
```

### Fusion: Probabilistic Decoupling
```python
# core/freecd.py
# Bayesian Probability Correction
T1_semantic = self.my_argmax(logits_scd[:1, :, :, :])
T2_semantic = self.my_argmax(logits_scd[1:, :, :, :])

# Adjust change mask by semantic consistency
logits = logits_bcd * (1 - (logits_scd[1:, :1, :, :]) * (logits_scd[:1, :1, :, :]))
```

---

## Inference

### Quick Start
```bash
python inference.py --dataset SECOND --backbone dinov2
```

### Batch Processing
```bash
python inference.py \
  --dataset SECOND \
  --backbone dinov2 \
  --t1-dir /path/to/T1 \
  --t2-dir /path/to/T2 \
  --output /path/to/output
```

### Output Structure
```
results/
├── cd/           # Binary change masks (white=change)
├── from/         # T1 original images, change regions only
├── to/           # T2 original images, change regions only
├── from_color/   # T1 semantic color map
└── to_color/     # T2 semantic color map
```

---

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| SECOND | 6 | Urban change detection |
| HIUCD | 8 | High-resolution urban change |
| HRSCD | 5 | High-resolution semantic change |
| JLSCD | 4 | Jilin-1 satellite change |
| LEVIR | 1 | Binary building change |
| WHU | 1 | Binary building change |
| LEVIRSCD | 15 | Fine-grained semantic change |

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{zhu2025freecd,
  title={Free-CD: Probabilistically Decoupled Training-Free Open-Vocabulary Change Detection with Resolution-Invariant Feature Inversion},
  author={Zhu, Yongshuo and Li, Lu and Chen, Keyan and Shi, Zhenwei and Zhou, Fugen},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Acknowledgement

This implementation is based on [FeatUp](https://github.com/mhamilton723/FeatUp) and [SegEarth](https://github.com/likyoo/SegEarth). Thanks for their excellent open-source work.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.