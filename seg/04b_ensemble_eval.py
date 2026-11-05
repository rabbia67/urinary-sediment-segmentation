"""
Average U-Net and SegFormer-B2 probabilities on full images.

The equal average uses both models on every class. The class-wise mix
chooses, on the validation images only, which model supplies each class
channel, then applies that choice to the test images.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "seg_eval", Path(__file__).resolve().parent / "04_seg_evaluation.py"
)
seg_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg_eval)

from utils.seg_data_utils import (
    CLASS_NAMES, DATA_ROOT, NUM_CLASSES, ORIGINAL_H, ORIGINAL_W,
    PATCH_SIZE, get_inference_patches, get_val_augmentation, load_split,
    stitch_patches,
)
from utils.seg_metrics import binary_dice, mean_dice

RESULTS_DIR = seg_eval.RESULTS_DIR
DEVICE = seg_eval.DEVICE


@torch.no_grad()
def predict_probs(model, image, transform) -> np.ndarray:
    """Return class probabilities, shape (C, H, W), cropped to the original image."""
    patches, grid_shape = get_inference_patches(image, PATCH_SIZE)
    batch = torch.stack([transform(image=patch)["image"] for patch in patches]).to(DEVICE)
    probs = torch.softmax(model(batch), dim=1).cpu().numpy().astype(np.float32)
    height, width = image.shape[:2]
    channels = []
    for cls_id in range(NUM_CLASSES):
        stitched = stitch_patches(
            [probs[i, cls_id] for i in range(len(patches))], grid_shape, PATCH_SIZE
        )
        channels.append(stitched[:height, :width])
    return np.stack(channels, axis=0)


def pooled_dice(preds, masks):
    return seg_eval.pooled_class_dice(preds, masks, NUM_CLASSES)


def summarise(name, preds, masks):
    mean_scores = [mean_dice(pred, gt, NUM_CLASSES) for pred, gt in zip(preds, masks)]
    binary_scores = [binary_dice(pred, gt) for pred, gt in zip(preds, masks)]
    pooled = pooled_dice(preds, masks)
    fg_pred = np.concatenate([(pred > 0).ravel() for pred in preds])
    fg_gt = np.concatenate([(gt > 0).ravel() for gt in masks])
    inter = int(np.logical_and(fg_pred, fg_gt).sum())
    pooled_binary = float(2 * inter / (int(fg_pred.sum()) + int(fg_gt.sum())))
    print(
        f"{name}: mean {np.mean(mean_scores):.4f} +/- {np.std(mean_scores):.4f}, "
        f"binary {np.mean(binary_scores):.4f} +/- {np.std(binary_scores):.4f}, "
        f"pooled binary {pooled_binary:.4f}"
    )
    return {
        "mean_dice": float(np.mean(mean_scores)),
        "std_dice": float(np.std(mean_scores)),
        "mean_binary_dice": float(np.mean(binary_scores)),
        "std_binary_dice": float(np.std(binary_scores)),
        "pooled_binary_dice": pooled_binary,
        "per_image_binary_dice": [float(v) for v in binary_scores],
        "pooled_dice": pooled,
    }


def class_owner(val_pooled):
    """Which model has the higher validation pooled Dice for each foreground class."""
    owner = {}
    for cls_id in range(1, NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        unet_dice = val_pooled["unet"][name]["dice"]
        sf_dice = val_pooled["segformer"][name]["dice"]
        owner[name] = "unet" if unet_dice >= sf_dice else "segformer"
        print(f"  {name}: validation U-Net {unet_dice:.4f}, SegFormer {sf_dice:.4f} -> {owner[name]}")
    return owner


def mix_probs(unet_p, sf_p, owner):
    mixed = np.empty_like(unet_p)
    mixed[0] = 0.5 * (unet_p[0] + sf_p[0])
    for cls_id in range(1, NUM_CLASSES):
        source = unet_p if owner[CLASS_NAMES[cls_id]] == "unet" else sf_p
        mixed[cls_id] = source[cls_id]
    mixed /= np.clip(mixed.sum(axis=0, keepdims=True), 1e-8, None)
    return mixed


def main():
    print("Loading models on", DEVICE)
    transform = get_val_augmentation()
    unet = seg_eval.load_model("unet")
    segformer = seg_eval.load_model("segformer")

    print("\nValidation pass")
    val_images, val_masks, _ = load_split(str(DATA_ROOT), "val")
    unet_val, sf_val = [], []
    for i, image in enumerate(val_images):
        unet_p = predict_probs(unet, image, transform)
        sf_p = predict_probs(segformer, image, transform)
        unet_val.append(unet_p.argmax(axis=0).astype(np.uint8))
        sf_val.append(sf_p.argmax(axis=0).astype(np.uint8))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(val_images)}")
    owner = class_owner({
        "unet": pooled_dice(unet_val, val_masks),
        "segformer": pooled_dice(sf_val, val_masks),
    })

    print("\nTest pass")
    test_images, test_masks, _ = load_split(str(DATA_ROOT), "test")
    equal_preds, mixed_preds = [], []
    for i, image in enumerate(test_images):
        unet_p = predict_probs(unet, image, transform)
        sf_p = predict_probs(segformer, image, transform)
        equal_preds.append((0.5 * (unet_p + sf_p)).argmax(axis=0).astype(np.uint8))
        mixed_preds.append(mix_probs(unet_p, sf_p, owner).argmax(axis=0).astype(np.uint8))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(test_images)}")

    results = {
        "equal_average": summarise("Equal average", equal_preds, test_masks),
        "validation_class_mix": summarise("Validation class mix", mixed_preds, test_masks),
        "class_owner_from_validation": owner,
    }
    out = RESULTS_DIR / "ensemble_evaluation.json"
    with open(out, "w") as handle:
        json.dump(results, handle)
    print("Saved", out)


if __name__ == "__main__":
    main()
