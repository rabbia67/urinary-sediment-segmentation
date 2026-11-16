# Urinary sediment segmentation

Multi-class segmentation of the public Urinary Sediment detection set, with an infection-burden score counted from the painted masks.

PASCAL VOC boxes are filled into eight-class masks. Where two boxes overlap, the smaller box is painted last. U-Net with a ResNet-34 encoder and SegFormer-B2 are trained on the same 512×512 patches. On a full image, probability fusion is the pixel-wise mean of the two softmax maps, followed by the argmax. The held-out scores below are agreement with those filled boxes.

The images are not in this repository.

## Dataset

| | |
|---|---|
| Images | 5,376 colour JPEG |
| Official split | 4,256 train / 852 validation / 268 test |
| Size | 5,370 images are 800×600. Six training or validation images are 768×576 or 768×568. Every test image is 800×600. |
| Boxes | PASCAL VOC |
| Classes | background, cast, crystal, epithelial cell, epithelial nucleus, erythrocyte, leukocyte, mycete |

Class list, VOC layout, and the official split: [174614361/Urinary-Sediment-Dataset](https://github.com/174614361/Urinary-Sediment-Dataset).

JPEG and XML files: [Google Drive folder](https://drive.google.com/drive/folders/18VqmoqK7dVSdxiyEE6qCfIpS8UOqqIBS).

Place the extracted release at `data/urinary-sediment/` two directories above this repository, so the scripts find `JPEGImages/`, `Annotations/`, and `ImageSets/Main/`.

| VOC name | Class | Index |
|---|---|---|
| — | Background | 0 |
| `cast` | Cast | 1 |
| `cryst` | Crystal | 2 |
| `epith` | Epithelial cell | 3 |
| `epithn` | Epithelial nucleus | 4 |
| `eryth` | Erythrocyte | 5 |
| `leuko` | Leukocyte | 6 |
| `mycete` | Mycete | 7 |

## Training

Patches are cut after the official image split. A patch whose foreground fraction is below 0.5% is discarded. The reported run kept 6,706 training patches (stride 256), 672 validation patches (stride 512), and 210 test patches (stride 512).

Both networks minimise `0.5 Dice + 0.5 focal`, with focal γ = 2. The class weight multiplies the cross-entropy of the true class. Training runs for at most 200 epochs, does not stop before epoch 100, and then stops after 30 epochs without a better validation mean Dice.

| | U-Net | SegFormer-B2 |
|---|---|---|
| Initialisation | ResNet-34, ImageNet | ADE20K, head replaced with 8 classes |
| Batch size | 8 | 2 |
| Learning rate | 1×10⁻⁴ | 6×10⁻⁵ |
| Schedule | Cosine annealing with warm restarts | 5-epoch linear warm-up, then polynomial decay (power 0.9) |
| Other | Encoder frozen through epoch 10, then unfrozen at epoch 11 and trained at 0.1× the learning rate. Weight decay 1×10⁻⁴. | Weight decay 1×10⁻⁴ |
| Parameters | 24,437,384 | 27,352,776 |
| Device | RTX 3090 | RTX 3090 |
| Selected checkpoint | Epoch 115 | Epoch 79 |

Mean Dice averages the seven foreground classes and scores a class that is absent from both the prediction and the image as 1. Binary Dice collapses every non-background label to one foreground class and does not give that credit.

## Infection-burden score

On each painted mask the score uses three area fractions: mycete, erythrocytes and leukocytes together, and epithelium (cell plus nucleus). Each fraction is mapped through its training-set mid-rank distribution. The weights are fixed:

```
S = 0.5 rank(mycete) + 0.3 rank(erythrocyte + leukocyte) + 0.2 rank(epithelium)
```

Tertiles use the 33.33 and 66.67 percentiles of `S` on the training images: 0.4663 and 0.5188. The same maps and cut-points are applied to validation and test. This score is not a clinical grade.

## Reproduce

Packages other than PyTorch are listed in `requirements-urinary-sediment.txt`. Training used PyTorch 2.14.0 with CUDA 12.6 on one RTX 3090.

```bash
python seg/prepare_urinary_sediment.py
python seg/01_seg_preprocessing.py
python seg/02_unet_training.py
python seg/03_segformer_training.py
python seg/04_seg_evaluation.py --test-only
python seg/04b_ensemble_eval.py
```

`prepare_urinary_sediment.py` writes `SegmentationClass/` beside the downloaded images. Preprocessing writes patch arrays, class weights, and the tertile labels under `outputs/seg_results/`. Training writes checkpoints under `outputs/seg_models/`. The two evaluation scripts expect `unet_best.pth` and `segformer_best.pth` in that directory.

Checkpoints, patch arrays, and the full-image prediction arrays are omitted from this repository because of size. The JSON summaries and the figures from the reported run are included.

## Held-out scores

Validation patches, from the saved training logs:

| Model | Stopped | Best mean Dice | Highest binary Dice |
|---|---|---|---|
| U-Net (ResNet-34) | epoch 145 | 0.9445 at epoch 115 | 0.8017 |
| SegFormer-B2 | epoch 109 | 0.9543 at epoch 79 | 0.8219 |

The U-Net log records binary Dice 0.7923 at epoch 115. The SegFormer log does not record binary Dice at epoch 79. The training summary stores the highest SegFormer binary Dice, 0.8219.

Full test images, one score per image. The ± term is the standard deviation across the 268 images.

| Model | Mean Dice | Binary Dice | Pooled binary Dice |
|---|---|---|---|
| U-Net (ResNet-34) | 0.9184 ± 0.0853 | 0.7654 ± 0.1843 | 0.7878 |
| SegFormer-B2 | 0.8721 ± 0.1065 | 0.7267 ± 0.1919 | 0.7730 |
| Probability fusion | 0.9192 ± 0.0824 | 0.7905 ± 0.1544 | 0.8117 |

On mean Dice, U-Net minus SegFormer-B2 is 0.0463 (bootstrap 95% interval 0.0311 to 0.0606; paired Wilcoxon p = 7.23×10⁻⁹). On binary Dice the same pairing gives p = 0.0065. Probability fusion minus U-Net is 0.0251 (p = 1.96×10⁻⁶). Probability fusion minus SegFormer-B2 is 0.0638 (p = 1.40×10⁻¹⁶).

Pooled Dice on the test images that contain each class. An absent class is not given a score of 1.

| Class | Images | U-Net | SegFormer-B2 | Probability fusion |
|---|---|---|---|---|
| Cast | 80 | 0.6871 | 0.6622 | 0.7179 |
| Crystal | 42 | 0.8224 | 0.6736 | 0.8685 |
| Epithelial cell | 75 | 0.8172 | 0.8339 | 0.8378 |
| Epithelial nucleus | 23 | 0.6934 | 0.7162 | 0.7221 |
| Erythrocyte | 48 | 0.8463 | 0.8604 | 0.8650 |
| Leukocyte | 54 | 0.7787 | 0.7787 | 0.8100 |
| Mycete | 39 | 0.8284 | 0.8169 | 0.8511 |

Figures from this run:

- `outputs/seg_figures/fig01_class_distribution.png`
- `outputs/seg_figures/fig02_severity_distribution.png`
- `outputs/seg_figures/fig03_sample_images.png`
- `outputs/seg_figures/fig04_unet_training_curves.png`
- `outputs/seg_figures/fig05_segformer_training_curves.png`
- `outputs/seg_figures/fig06_dice_comparison.png`
- `outputs/seg_figures/fig07_qualitative_comparison.png`
- `outputs/seg_figures/fig08_seg_confusion_matrices.png`

Numeric summaries: `outputs/seg_results/segmentation_evaluation.json`, `outputs/seg_results/ensemble_evaluation.json`, and `outputs/seg_results/segmentation_metrics.csv`.

## Layout

| Path | Role |
|---|---|
| `seg/prepare_urinary_sediment.py` | Paint VOC boxes into masks |
| `seg/01_seg_preprocessing.py` | Patches, class weights, tertile labels |
| `seg/02_unet_training.py` | U-Net training |
| `seg/03_segformer_training.py` | SegFormer-B2 training |
| `seg/04_seg_evaluation.py` | Full-image test scores |
| `seg/04b_ensemble_eval.py` | Probability fusion |
| `seg/models/losses.py` | Dice and focal loss |
| `utils/seg_data_utils.py` | Patches, augmentation, class weights |
| `utils/seg_metrics.py` | Dice and the per-class scores |

## License

MIT. See [LICENSE](LICENSE).
