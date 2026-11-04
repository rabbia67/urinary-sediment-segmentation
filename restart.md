# SegFormer on the RTX 3090

The 3090 continuation already finished. Training stopped at epoch 109. The best checkpoint is epoch 79, validation mean Dice 0.9543, binary Dice at that checkpoint 0.8189. The highest binary Dice on the run is 0.8219. U-Net had already finished at epoch 145, best mean Dice 0.9445 at epoch 115. Checkpoints are not stored in git.

The notes below are the command that was used to resume from the earlier epoch-18 checkpoint. They are not a request to train again.

## Copy these

From this project directory:

- `seg/03_segformer_training.py` and the rest of the code
- `outputs/seg_models/segformer_best.pth` (same weights as `outputs/seg_models/for_3090/segformer_epoch18.pth`)
- `outputs/seg_results/patches_train_images.npy`
- `outputs/seg_results/patches_train_masks.npy`
- `outputs/seg_results/patches_val_images.npy`
- `outputs/seg_results/patches_val_masks.npy`
- `outputs/seg_results/class_weights.npy`
- `../wheels` if those packages are not already installed

Torch 2.14.0+cu126 and torchvision are already on the 3090. Do not reinstall them. The wheel folder does not contain either package.

The first launch also needs the Hugging Face snapshot `nvidia/segformer-b2-finetuned-ade-512-512`. Copy `~/.cache/huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512`, or let the script download it.

## Install the other packages

From the project directory, with the 3090 environment activated:

```bash
pip install --no-index --find-links ../wheels -r requirements-urinary-sediment.txt
```

## Start from the checkpoint

Use the 3090 directly. Do not set `CUDA_VISIBLE_DEVICES=1`.

```bash
cd uti-prediction-urinary-sediment
mkdir -p outputs/seg_results outputs/seg_figures outputs/seg_models
python -u seg/03_segformer_training.py \
  --resume outputs/seg_models/segformer_best.pth \
  2>&1 | tee outputs/seg_results/segformer_3090.log
```

The script loads that checkpoint and continues at epoch 19. It still runs to a maximum of 200 epochs and will not early-stop before epoch 100. After each epoch it writes:

- `outputs/seg_models/segformer_latest.pth`
- `outputs/seg_models/segformer_best.pth` when the validation Dice improves

`BATCH_SIZE` in `seg/03_segformer_training.py` is 2 because the 8 GB card ran out of memory at 4. On the 24 GB 3090, raise that number before launching if you want fewer steps per epoch. The saved optimizer state was created at batch size 2.
