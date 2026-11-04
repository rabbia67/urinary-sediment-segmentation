# Urinary sediment segmentation

Code for the multi-class segmentation experiments on the public Urinary Sediment detection set. PASCAL VOC boxes are painted into eight-class masks, with the smaller box kept where two boxes overlap. U-Net with a ResNet-34 encoder and SegFormer-B2 are trained on 512×512 patches.

## Dataset

The images are not in this repository.

- Class list, VOC layout, and official split: https://github.com/174614361/Urinary-Sediment-Dataset
- JPEG and XML files: https://drive.google.com/drive/folders/18VqmoqK7dVSdxiyEE6qCfIpS8UOqqIBS

The release has 5,376 colour JPEGs of 800×600, with 4,256 training, 852 validation, and 268 test images. The seven box classes are cast, crystal (`cryst`), epithelial cell (`epith`), epithelial nucleus (`epithn`), erythrocyte (`eryth`), leukocyte (`leuko`), and mycete.

This checkout expects the extracted folders at `../../data/urinary-sediment/` relative to the repository root (`JPEGImages/`, `Annotations/`, `ImageSets/Main/`). That is the layout used for the reported run.

## Run

```bash
python seg/prepare_urinary_sediment.py
python seg/01_seg_preprocessing.py
python seg/02_unet_training.py
python seg/03_segformer_training.py
```

Training uses a maximum of 200 epochs, patience 30, and does not stop before epoch 100. U-Net uses batch size 8. SegFormer uses batch size 2. Packages other than PyTorch are listed in `requirements-urinary-sediment.txt`. The U-Net machine used the CUDA 13 build in `requirements-torch-cu130.txt`. The SegFormer continuation used PyTorch 2.14.0 with CUDA 12.6.

## Validation-patch scores from the reported run

Mean Dice averages the seven foreground classes and scores a class that is absent from the patch as 1. Binary Dice is the foreground-versus-background score. Both are agreement with filled boxes on validation patches.

| Model | Stopped | Best mean Dice | Highest binary Dice |
|---|---|---|---|
| U-Net (ResNet-34) | epoch 145 | 0.9445 at epoch 115 | 0.8017 |
| SegFormer-B2 | epoch 109 | 0.9543 at epoch 79 | 0.8219 |

Summary JSON files and the training figures are under `outputs/seg_results/` and `outputs/seg_figures/`. Checkpoints and patch arrays are omitted because of size.

## License

MIT. See [LICENSE](LICENSE).
