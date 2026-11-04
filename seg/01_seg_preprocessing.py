"""
01_seg_preprocessing.py
========================
Urinary Sediment Dataset — Data Preparation

This script:
1. Loads the prepared JPEGs (800x600) and box-derived masks (values 0-7)
2. Computes per-class pixel/object statistics across all splits
3. Derives severity labels from GT masks using percentile thresholds (train set only)
4. Extracts 512x512 patches (stride 256 for training, 512 for inference)
5. Saves class_statistics.json, severity_labels.json, severity_thresholds.json, EDA figures

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.seg_data_utils import (
    load_split, compute_class_statistics, extract_patches,
    CLASS_NAMES, NUM_CLASSES, PATCH_SIZE, DATA_ROOT, ORIGINAL_H, ORIGINAL_W
)

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def extract_severity_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Extract severity-relevant features from a single GT mask.

    Returns normalized feature values for severity scoring.
    """
    total_pixels = mask.size
    foreground = (mask > 0).sum()

    # Classes: 7 mycete, 5+6 red and white cells, 2 crystals, 3+4 epithelial.
    mycete_area = (mask == 7).sum() / total_pixels
    inflammation = ((mask == 5).sum() + (mask == 6).sum()) / total_pixels
    crystal_area = (mask == 2).sum() / total_pixels
    epithelial_total = ((mask == 3).sum() + (mask == 4).sum()) / total_pixels

    return {
        'mycete_area': mycete_area,
        'epithelial_total': epithelial_total,
        'inflammation': inflammation,
        'foreground_ratio': foreground / total_pixels,
        'crystal_area': crystal_area,
    }


def _component_matrix(features_list: List[Dict[str, float]]) -> np.ndarray:
    """Columns are mycete area, red/white-cell area, and epithelial area."""
    return np.column_stack([
        [f['mycete_area'] for f in features_list],
        [f['inflammation'] for f in features_list],
        [f['epithelial_total'] for f in features_list],
    ]).astype(np.float64)


def _train_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values through the training-set mid-rank CDF. No other split enters the ranks."""
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(reference)
    if n == 0 or np.isclose(reference.max(), reference.min()):
        return np.zeros_like(values)
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side='left')
    right = np.searchsorted(ordered, values, side='right')
    return (left + right) / (2.0 * n)


def score_from_train(features_list: List[Dict[str, float]],
                     train_features: List[Dict[str, float]]) -> np.ndarray:
    """
    S = 0.5 * rank(mycete area) + 0.3 * rank(red/white-cell area) + 0.2 * rank(epithelial area).
    Rank maps are estimated on the training images only.
    """
    values = _component_matrix(features_list)
    reference = _component_matrix(train_features)
    return (0.5 * _train_cdf(reference[:, 0], values[:, 0]) +
            0.3 * _train_cdf(reference[:, 1], values[:, 1]) +
            0.2 * _train_cdf(reference[:, 2], values[:, 2]))


def derive_severity_labels(train_features: List[Dict], all_features: Dict[str, List[Dict]],
                           all_filenames: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
    """
    Derive 3-class severity labels using percentile thresholds from training set.

    Parameters
    ----------
    train_features : list of feature dicts from training masks
    all_features : {'train': [...], 'val': [...], 'test': [...]}
    all_filenames : {'train': [...], 'val': [...], 'test': [...]}

    Returns
    -------
    severity_labels : {filename: label} for all images
    thresholds : {p33: float, p66: float}
    """
    # Thresholds and rank maps use the training images only.
    train_scores = score_from_train(train_features, train_features)

    p33 = float(np.percentile(train_scores, 33.33))
    p66 = float(np.percentile(train_scores, 66.67))

    thresholds = {'p33': p33, 'p66': p66, 'rank_scope': 'train_only'}

    severity_labels = {}

    for split_name, features_list in all_features.items():
        scores = score_from_train(features_list, train_features)
        filenames = all_filenames[split_name]

        for fname, score in zip(filenames, scores):
            # Use split-prefixed key to avoid collision (all splits share 0001-0100.tif)
            key = f"{split_name}/{fname}"
            if score < p33:
                label = 0  # Mild
            elif score < p66:
                label = 1  # Moderate
            else:
                label = 2  # Severe

            severity_labels[key] = {
                'label': int(label),
                'severity': ['mild', 'moderate', 'severe'][label],
                'score': float(score),
                'split': split_name,
                'filename': fname,
            }

    return severity_labels, thresholds


def plot_class_distribution(stats: Dict, save_path: Path):
    """Plot per-class pixel distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]  # Exclude background
    pixel_counts = [stats[c]['total_pixels'] for c in classes]
    object_counts = [stats[c]['total_objects'] for c in classes]

    colors = sns.color_palette("Set2", len(classes))

    # Pixel distribution
    bars = axes[0].barh(classes, pixel_counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Total Pixels')
    axes[0].set_title('Per-Class Pixel Distribution')
    for bar, val in zip(bars, pixel_counts):
        axes[0].text(bar.get_width() + max(pixel_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', va='center', fontsize=8)

    # Object count distribution
    bars = axes[1].barh(classes, object_counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Total Objects')
    axes[1].set_title('Per-Class Object Count Distribution')
    for bar, val in zip(bars, object_counts):
        axes[1].text(bar.get_width() + max(object_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', va='center', fontsize=8)

    plt.suptitle('Urinary Sediment Dataset: Class Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_severity_distribution(severity_labels: Dict, save_path: Path):
    """Plot severity label distribution per split."""
    splits = ['train', 'val', 'test']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    severity_names = ['Mild', 'Moderate', 'Severe']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    for ax, split in zip(axes, splits):
        labels = [v['label'] for v in severity_labels.values() if v['split'] == split]
        counts = [labels.count(i) for i in range(3)]

        bars = ax.bar(severity_names, counts, color=colors, edgecolor='black')
        n_total = max(len(labels), 1)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{count}\n({count / n_total * 100:.0f}%)',
                    ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{split.capitalize()} (n={len(labels)})')
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(counts) * 1.3)

    plt.suptitle('Severity Label Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_images(images: List[np.ndarray], masks: List[np.ndarray],
                       filenames: List[str], save_path: Path, n_samples: int = 3):
    """Plot sample images with their multi-class masks."""
    indices = np.linspace(0, len(images) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 8))

    # Color map for classes
    cmap_colors = np.array([
        [0, 0, 0],       # 0: Background - black
        [255, 0, 0],     # 1: Cast
        [0, 255, 0],     # 2: Crystal
        [0, 0, 255],     # 3: Epithelial cell
        [255, 255, 0],   # 4: Epithelial nucleus
        [255, 0, 255],   # 5: Erythrocyte
        [0, 255, 255],   # 6: Leukocyte
        [128, 128, 0],   # 7: Mycete
    ], dtype=np.uint8)

    for col, idx in enumerate(indices):
        # Image
        axes[0, col].imshow(images[idx])
        axes[0, col].set_title(f'{filenames[idx]}', fontsize=10)
        axes[0, col].axis('off')

        # Mask (colored)
        mask_colored = cmap_colors[masks[idx]]
        axes[1, col].imshow(mask_colored)
        axes[1, col].set_title('Multi-class Mask', fontsize=10)
        axes[1, col].axis('off')

    plt.suptitle('Sample Images and Ground Truth Masks', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("URINARY SEDIMENT PREPROCESSING")
    print("=" * 60)

    # Step 1: Load all splits
    print("\n[Step 1] Loading images and masks...")
    all_images = {}
    all_masks = {}
    all_filenames = {}

    for split in ['train', 'val', 'test']:
        print(f"  Loading {split}...")
        images, masks, fnames = load_split(str(DATA_ROOT), split)
        all_images[split] = images
        all_masks[split] = masks
        all_filenames[split] = fnames
        print(f"    Loaded {len(images)} images, shape: {images[0].shape}")

    # Step 2: Compute class statistics
    print("\n[Step 2] Computing class statistics...")
    all_masks_flat = all_masks['train'] + all_masks['val'] + all_masks['test']
    class_stats = compute_class_statistics(all_masks_flat)

    print("\n  Class Statistics:")
    for cls_name, stats in class_stats.items():
        if stats['class_id'] > 0:
            print(f"    {cls_name}: {stats['total_objects']} objects, "
                  f"{stats['total_pixels']:,} pixels, "
                  f"in {stats['n_images_present']}/{len(all_masks_flat)} images")

    with open(RESULTS_DIR / 'class_statistics.json', 'w') as f:
        json.dump(class_stats, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'class_statistics.json'}")

    # Step 3: Derive severity labels
    print("\n[Step 3] Deriving severity labels...")

    all_features = {}
    for split in ['train', 'val', 'test']:
        all_features[split] = [extract_severity_features(m) for m in all_masks[split]]

    severity_labels, thresholds = derive_severity_labels(
        all_features['train'], all_features, all_filenames
    )

    # Print distribution
    for split in ['train', 'val', 'test']:
        labels = [v['label'] for v in severity_labels.values() if v['split'] == split]
        counts = [labels.count(i) for i in range(3)]
        print(f"  {split}: Mild={counts[0]}, Moderate={counts[1]}, Severe={counts[2]}")

    with open(RESULTS_DIR / 'severity_labels.json', 'w') as f:
        json.dump(severity_labels, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'severity_labels.json'}")

    with open(RESULTS_DIR / 'severity_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'severity_thresholds.json'}")

    # Step 4: Extract patches
    print("\n[Step 4] Extracting patches...")
    patch_counts = {}

    for split in ['train', 'val', 'test']:
        stride = 256 if split == 'train' else PATCH_SIZE
        patches = []

        for img, msk in zip(all_images[split], all_masks[split]):
            img_patches = extract_patches(img, msk, PATCH_SIZE, stride)
            patches.extend(img_patches)

        patch_counts[split] = len(patches)
        print(f"  {split}: {len(patches)} patches (stride={stride})")

        # Save patches as numpy arrays
        patch_images = np.array([p[0] for p in patches], dtype=np.uint8)
        patch_masks = np.array([p[1] for p in patches], dtype=np.uint8)

        np.save(RESULTS_DIR / f'patches_{split}_images.npy', patch_images)
        np.save(RESULTS_DIR / f'patches_{split}_masks.npy', patch_masks)
        print(f"    Saved: patches_{split}_images.npy ({patch_images.shape})")

    # Step 5: Compute class weights for training
    print("\n[Step 5] Computing class weights...")
    from utils.seg_data_utils import compute_class_weights
    class_weights = compute_class_weights(all_masks['train'])
    np.save(RESULTS_DIR / 'class_weights.npy', class_weights)
    print(f"  Class weights: {dict(zip(CLASS_NAMES.values(), class_weights.tolist()))}")

    # Step 6: Generate EDA figures
    print("\n[Step 6] Generating EDA figures...")
    plot_class_distribution(class_stats, FIGURES_DIR / 'fig01_class_distribution.png')
    plot_severity_distribution(severity_labels, FIGURES_DIR / 'fig02_severity_distribution.png')
    plot_sample_images(all_images['train'], all_masks['train'],
                       all_filenames['train'], FIGURES_DIR / 'fig03_sample_images.png')

    # Save dataset summary
    summary = {
        'total_images': sum(len(v) for v in all_filenames.values()),
        'splits': {name: len(files) for name, files in all_filenames.items()},
        'image_size': {'height': ORIGINAL_H, 'width': ORIGINAL_W},
        'mask_source': 'VOC bounding boxes rasterized with smaller boxes on top',
        'patch_size': PATCH_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'patches_per_split': patch_counts,
        'severity_thresholds': thresholds,
        'class_weights': class_weights.tolist(),
    }

    with open(RESULTS_DIR / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
