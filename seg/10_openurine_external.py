"""
10_openurine_external.py
========================
Zero-shot external evaluation on the OpenUrine collection (Alizadeh et al., 2025).

No training happens here. The U-Net and SegFormer-B2 checkpoints selected on the
Liang validation patches (outputs/seg_models/unet_best.pth, segformer_best.pth)
are applied as they are, and probability fusion is the pixel-wise mean of the
two softmax maps, exactly as in 04b_ensemble_eval.py.

OpenUrine ships YOLO boxes (class cx cy w h, normalised) for 39 classes. The
39 names are mapped onto the paper's seven foreground classes (Table MAP39).
Boxes of a class with no counterpart (bacteria, sperm, mucus, parasite eggs,
artifacts, fat, atypical cells, Trichomonas) receive the IGNORE label 255 and
are excluded from every Dice. A second reference, in which those pixels count
as background, is also scored and saved.

Images span 534x401 to 4032x3024 pixels, so each image is resized before
inference. The default rule scales the longer side to TARGET_LONG px, and the
script runs every value in --long-sides so the paper can report a particle-
matched scale and a frame-matched scale. The boxes are filled at the resized
resolution, larger box first, so a smaller box keeps the pixel.

Usage (on the machine that holds the checkpoints):
    python seg/10_openurine_external.py --openurine D:/UTI/OpenUrine_extract/OpenUrine
    python seg/10_openurine_external.py --openurine ... --long-sides 1600 800
    python seg/10_openurine_external.py --openurine ... --dry-run   # plumbing test, random model

Outputs under outputs/seg_results/:
    openurine_external.json      summary per scale, per model, per reference variant
    openurine_per_image_<L>.csv  per-image binary Dice and area fractions
    openurine_table_<L>.tex      LaTeX rows for the manuscript table
If the Liang training masks are reachable at DATA_ROOT, the infection-burden
score is also computed with the training rank maps and the stored tertiles.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.seg_data_utils import CLASS_NAMES, DATA_ROOT, NUM_CLASSES  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
IGNORE = 255

OPENURINE_NAMES = [
    'Ammonium Biurat', 'Amorphous', 'Artifact', 'Bacteria', 'Bilirubin', 'Calcium Carbonate',
    'Calcium Oxalate', 'Calcium Phosphate', 'Cholesterol', 'Cystine', 'Enterobius vermicularis Egg',
    'Epithelial Cell -Renal-', 'Epithelial Cell -Squamous-', 'Epithelial Cell -Transitional-',
    'Fat Droplets', 'Fungal Hyphae', 'Granular Cast', 'Hippuric Acid', 'Hyaline Cast', 'Leucine',
    'Lipid Cast', 'Mixed Cell Cast', 'Mucus', 'Muddy Brown Cast', 'Oval Fat Bodies', 'RBC', 'RBC Cast',
    'RBC Clump', 'Schistosoma haematobium Eggs', 'Sperm', 'Suspected atypical cell',
    'Trichomonas vaginalis', 'Triple Phosphate', 'Tyrosine crystal', 'Uric Acid', 'WBC', 'WBC Clump',
    'Waxy Cast', 'Yeast',
]

# OpenUrine name -> paper class id (1 cast, 2 crystal, 3 epithelial cell, 5 erythrocyte,
# 6 leukocyte, 7 mycete). Class 4 (epithelial nucleus) has no OpenUrine counterpart.
MAP39 = {
    'Granular Cast': 1, 'Hyaline Cast': 1, 'Lipid Cast': 1, 'Mixed Cell Cast': 1,
    'Muddy Brown Cast': 1, 'RBC Cast': 1, 'Waxy Cast': 1,
    'Ammonium Biurat': 2, 'Amorphous': 2, 'Bilirubin': 2, 'Calcium Carbonate': 2,
    'Calcium Oxalate': 2, 'Calcium Phosphate': 2, 'Cholesterol': 2, 'Cystine': 2,
    'Hippuric Acid': 2, 'Leucine': 2, 'Triple Phosphate': 2, 'Tyrosine crystal': 2, 'Uric Acid': 2,
    'Epithelial Cell -Renal-': 3, 'Epithelial Cell -Squamous-': 3, 'Epithelial Cell -Transitional-': 3,
    'RBC': 5, 'RBC Clump': 5,
    'WBC': 6, 'WBC Clump': 6,
    'Fungal Hyphae': 7, 'Yeast': 7,
}
IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


# --------------------------------------------------------------------------- data
def list_stems(data_dir: Path) -> list[tuple[str, Path, Path]]:
    items = []
    for label in sorted(data_dir.glob("*.txt")):
        image = next((p for p in data_dir.glob(label.stem + ".*") if p.suffix.lower() in IMAGE_EXT), None)
        if image is not None:
            items.append((label.stem, image, label))
    return items


def read_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def resize_image(image: Image.Image, long_side: int) -> tuple[np.ndarray, float]:
    w, h = image.size
    scale = long_side / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    arr = np.array(image.convert("RGB").resize((new_w, new_h), resample), dtype=np.uint8)
    return arr, scale


def boxes_to_masks(boxes, height: int, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mask with IGNORE for unmapped boxes, mask with unmapped as background, n_unmapped)."""
    painted = []
    n_unmapped = 0
    for cid, cx, cy, bw, bh in boxes:
        name = OPENURINE_NAMES[cid]
        target = MAP39.get(name)
        x1 = max(0, int(round((cx - bw / 2) * width)))
        x2 = min(width, int(round((cx + bw / 2) * width)))
        y1 = max(0, int(round((cy - bh / 2) * height)))
        y2 = min(height, int(round((cy + bh / 2) * height)))
        if x2 <= x1 or y2 <= y1:
            continue
        if target is None:
            n_unmapped += 1
            painted.append(((x2 - x1) * (y2 - y1), IGNORE, y1, y2, x1, x2))
        else:
            painted.append(((x2 - x1) * (y2 - y1), target, y1, y2, x1, x2))
    mask_ignore = np.zeros((height, width), dtype=np.uint8)
    mask_bg = np.zeros((height, width), dtype=np.uint8)
    # Larger boxes first, so a smaller box keeps the pixel (same rule as prepare_urinary_sediment.py).
    for _area, cls, y1, y2, x1, x2 in sorted(painted, reverse=True):
        mask_ignore[y1:y2, x1:x2] = cls
        if cls != IGNORE:
            mask_bg[y1:y2, x1:x2] = cls
    # In the background variant an unmapped box must not erase a mapped box it covers,
    # so mapped boxes were painted into mask_bg only; mask_ignore keeps the ignore rule.
    return mask_ignore, mask_bg, n_unmapped


# --------------------------------------------------------------------------- metrics
def dice_binary(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p = (pred > 0) & valid
    g = (gt > 0) & valid
    denom = int(p.sum()) + int(g.sum())
    return float(2 * int((p & g).sum()) / denom) if denom else float("nan")


def mean_dice_image(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Average over the seven foreground classes; a class absent from both scores 1 (paper rule)."""
    scores = []
    for cls in range(1, NUM_CLASSES):
        p = (pred == cls) & valid
        g = (gt == cls) & valid
        denom = int(p.sum()) + int(g.sum())
        scores.append(1.0 if denom == 0 else 2 * int((p & g).sum()) / denom)
    return float(np.mean(scores))


class PooledDice:
    def __init__(self):
        self.inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.pred = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.gt = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.present = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.fg_inter = 0
        self.fg_pred = 0
        self.fg_gt = 0

    def add(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
        for cls in range(NUM_CLASSES):
            p = (pred == cls) & valid
            g = (gt == cls) & valid
            self.inter[cls] += int((p & g).sum())
            self.pred[cls] += int(p.sum())
            self.gt[cls] += int(g.sum())
            self.present[cls] += int(g.any())
        p = (pred > 0) & valid
        g = (gt > 0) & valid
        self.fg_inter += int((p & g).sum())
        self.fg_pred += int(p.sum())
        self.fg_gt += int(g.sum())

    def summary(self) -> dict:
        out = {}
        for cls in range(NUM_CLASSES):
            denom = int(self.pred[cls] + self.gt[cls])
            out[CLASS_NAMES[cls]] = {
                "dice": (2.0 * self.inter[cls] / denom) if denom else None,
                "intersection": int(self.inter[cls]),
                "pred_pixels": int(self.pred[cls]),
                "gt_pixels": int(self.gt[cls]),
                "images_present": int(self.present[cls]),
            }
        denom = self.fg_pred + self.fg_gt
        out["pooled_binary_dice"] = (2.0 * self.fg_inter / denom) if denom else None
        return out


# --------------------------------------------------------------------------- score
def area_fractions(mask: np.ndarray, valid: np.ndarray) -> dict:
    total = max(int(valid.sum()), 1)
    return {
        "mycete_area": float(((mask == 7) & valid).sum() / total),
        "inflammation": float((((mask == 5) | (mask == 6)) & valid).sum() / total),
        "epithelial_total": float((((mask == 3) | (mask == 4)) & valid).sum() / total),
    }


def load_training_reference():
    """Training-set area fractions for the rank maps, if the Liang masks are reachable."""
    mask_dir = DATA_ROOT / "SegmentationClass"
    list_path = DATA_ROOT / "ImageSets" / "Main" / "train.txt"
    if not (mask_dir.exists() and list_path.exists()):
        return None
    ref = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stem = Path(line.strip()).stem
        if not stem:
            continue
        mask = np.array(Image.open(mask_dir / f"{stem}.png"))
        ref.append(area_fractions(mask, np.ones(mask.shape, dtype=bool)))
    return ref


def train_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(reference)
    if n == 0 or np.isclose(reference.max(), reference.min()):
        return np.zeros_like(values)
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right) / (2.0 * n)


def scores_from_reference(feats: list[dict], ref: list[dict]) -> np.ndarray:
    cols = ("mycete_area", "inflammation", "epithelial_total")
    v = np.array([[f[c] for c in cols] for f in feats])
    r = np.array([[f[c] for c in cols] for f in ref])
    return 0.5 * train_cdf(r[:, 0], v[:, 0]) + 0.3 * train_cdf(r[:, 1], v[:, 1]) + 0.2 * train_cdf(r[:, 2], v[:, 2])


def band(score: float, p33: float, p66: float) -> str:
    return "severe" if score >= p66 else ("moderate" if score >= p33 else "mild")


# --------------------------------------------------------------------------- models
class RandomPredictor:
    """Dry-run stand-in: returns uniform-ish random probabilities. Plumbing test only."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        logits = self.rng.normal(size=(NUM_CLASSES, h, w)).astype(np.float32)
        logits[0] += 2.0
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)


def real_predictors():
    spec = importlib.util.spec_from_file_location("seg_eval", Path(__file__).resolve().parent / "04_seg_evaluation.py")
    seg_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seg_eval)
    spec_b = importlib.util.spec_from_file_location("seg_ens", Path(__file__).resolve().parent / "04b_ensemble_eval.py")
    seg_ens = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(seg_ens)
    from utils.seg_data_utils import get_val_augmentation
    transform = get_val_augmentation()
    unet = seg_eval.load_model("unet")
    segformer = seg_eval.load_model("segformer")
    print("Models loaded on", seg_eval.DEVICE)
    return (lambda img: seg_ens.predict_probs(unet, img, transform),
            lambda img: seg_ens.predict_probs(segformer, img, transform))


# --------------------------------------------------------------------------- main
def run_scale(items, long_side: int, predict_unet, predict_sf, ref_feats, thresholds, limit: int | None):
    variants = ("ignore_unmapped", "unmapped_as_background")
    models = ("unet", "segformer", "fusion")
    pooled = {v: {m: PooledDice() for m in models} for v in variants}
    per_image_mean = {v: {m: [] for m in models} for v in variants}
    per_image_bin = {v: {m: [] for m in models} for v in variants}
    rows = []
    gt_feats, pred_feats = [], {m: [] for m in models}
    n_unmapped_total = 0
    stems_with_mapped = 0

    for idx, (stem, image_path, label_path) in enumerate(items[:limit] if limit else items):
        with Image.open(image_path) as im:
            image, scale = resize_image(im, long_side)
        h, w = image.shape[:2]
        gt_ignore, gt_bg, n_unmapped = boxes_to_masks(read_yolo(label_path), h, w)
        n_unmapped_total += n_unmapped
        if (gt_bg > 0).any():
            stems_with_mapped += 1
        p_unet = predict_unet(image)
        p_sf = predict_sf(image)
        preds = {
            "unet": p_unet.argmax(axis=0).astype(np.uint8),
            "segformer": p_sf.argmax(axis=0).astype(np.uint8),
            "fusion": (0.5 * (p_unet + p_sf)).argmax(axis=0).astype(np.uint8),
        }
        all_valid = np.ones((h, w), dtype=bool)
        valid_ignore = gt_ignore != IGNORE
        row = {"stem": stem, "group": "ex" if stem.startswith("ex") else "numeric",
               "native_w": im.size[0], "native_h": im.size[1], "scale": round(scale, 4),
               "resized_w": w, "resized_h": h, "n_unmapped_boxes": n_unmapped,
               "classes_present": "|".join(CLASS_NAMES[c] for c in np.unique(gt_bg) if c > 0)}
        for v, gt, valid in (("ignore_unmapped", gt_ignore, valid_ignore), ("unmapped_as_background", gt_bg, all_valid)):
            for m in models:
                pooled[v][m].add(preds[m], gt, valid)
                md = mean_dice_image(preds[m], gt, valid)
                bd = dice_binary(preds[m], gt, valid)
                per_image_mean[v][m].append(md)
                per_image_bin[v][m].append(bd)
                row[f"{m}_mean_dice_{v}"] = md
                row[f"{m}_binary_dice_{v}"] = bd
        gt_feats.append(area_fractions(gt_ignore, valid_ignore))
        for m in models:
            pred_feats[m].append(area_fractions(preds[m], valid_ignore))
        rows.append(row)
        if (idx + 1) % 50 == 0:
            print(f"  long side {long_side}: {idx + 1}/{len(items)}")

    summary = {"long_side": long_side, "n_images": len(rows), "n_images_with_mapped_class": stems_with_mapped,
               "n_unmapped_boxes_ignored": n_unmapped_total, "variants": {}}
    for v in variants:
        summary["variants"][v] = {}
        for m in models:
            md = np.array(per_image_mean[v][m], dtype=np.float64)
            bd = np.array(per_image_bin[v][m], dtype=np.float64)
            summary["variants"][v][m] = {
                "mean_dice": float(np.nanmean(md)), "std_dice": float(np.nanstd(md)),
                "mean_binary_dice": float(np.nanmean(bd)), "std_binary_dice": float(np.nanstd(bd)),
                "n_binary_scored": int(np.isfinite(bd).sum()),
                "pooled": pooled[v][m].summary(),
            }

    # Infection-burden score with the training rank maps and the stored tertiles.
    if ref_feats is not None:
        p33, p66 = thresholds["p33"], thresholds["p66"]
        gt_scores = scores_from_reference(gt_feats, ref_feats)
        summary["score"] = {"thresholds": thresholds, "reference": {"n_train": len(ref_feats)},
                            "gt_bands": dict(Counter(band(s, p33, p66) for s in gt_scores))}
        for i, s in enumerate(gt_scores):
            rows[i]["gt_score"] = float(s)
            rows[i]["gt_band"] = band(s, p33, p66)
        for m in models:
            ps = scores_from_reference(pred_feats[m], ref_feats)
            agree = np.mean([band(a, p33, p66) == band(b, p33, p66) for a, b in zip(gt_scores, ps)])
            summary["score"][m] = {"bands": dict(Counter(band(s, p33, p66) for s in ps)),
                                   "band_agreement_with_gt": float(agree),
                                   "spearman_with_gt": float(_spearman(gt_scores, ps))}
            for i, s in enumerate(ps):
                rows[i][f"{m}_score"] = float(s)
    else:
        summary["score"] = "not computed: Liang training masks not found at DATA_ROOT"
        for i, f in enumerate(gt_feats):
            rows[i].update({f"gt_{k}": v for k, v in f.items()})
    return summary, rows


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def write_tex_rows(summary: dict, path: Path):
    v = summary["variants"]["ignore_unmapped"]
    lines = ["% rows for the OpenUrine table; generated by 10_openurine_external.py",
             f"% long side {summary['long_side']} px, unmapped boxes ignored, n = {summary['n_images']}"]
    for cls in range(1, NUM_CLASSES):
        name = CLASS_NAMES[cls]
        cells = []
        for m in ("unet", "segformer", "fusion"):
            d = v[m]["pooled"][name]["dice"]
            cells.append("--" if d is None else f"{d:.4f}")
        n = v["unet"]["pooled"][name]["images_present"]
        lines.append(f"{name} & {n} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for m, label in (("unet", "U-Net (ResNet-34)"), ("segformer", "SegFormer-B2"), ("fusion", "Probability fusion")):
        s = v[m]
        lines.append(f"% {label}: mean Dice {s['mean_dice']:.4f} +/- {s['std_dice']:.4f}, "
                     f"binary {s['mean_binary_dice']:.4f} +/- {s['std_binary_dice']:.4f}, "
                     f"pooled binary {s['pooled']['pooled_binary_dice']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openurine", type=Path, required=True, help="Folder holding data.yaml and Data/")
    ap.add_argument("--long-sides", type=int, nargs="+", default=[1600, 800],
                    help="Longer-side sizes to evaluate; 1600 matches the Liang erythrocyte scale, 800 the Liang frame width")
    ap.add_argument("--dry-run", action="store_true", help="Random predictor, no checkpoints needed")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    args = ap.parse_args()

    data_dir = args.openurine / "Data"
    items = list_stems(data_dir)
    if not items:
        raise SystemExit(f"No image/label pairs under {data_dir}")
    print(f"OpenUrine: {len(items)} image/label pairs")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((RESULTS_DIR / "severity_thresholds.json").read_text())
    ref_feats = load_training_reference()
    print("Training rank maps:", "available" if ref_feats else "not available (score skipped)")

    if args.dry_run:
        predict_unet, predict_sf = RandomPredictor(0), RandomPredictor(1)
    else:
        predict_unet, predict_sf = real_predictors()

    results = {"class_map": MAP39,
               "unmapped_classes": [n for n in OPENURINE_NAMES if n not in MAP39],
               "dry_run": args.dry_run, "scales": {}}
    for long_side in args.long_sides:
        print(f"\nScale: longer side -> {long_side} px")
        summary, rows = run_scale(items, long_side, predict_unet, predict_sf, ref_feats, thresholds, args.limit)
        results["scales"][str(long_side)] = summary
        with open(RESULTS_DIR / f"openurine_per_image_{long_side}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "stem", k)))
            writer.writeheader()
            writer.writerows(rows)
        write_tex_rows(summary, RESULTS_DIR / f"openurine_table_{long_side}.tex")
        v = summary["variants"]["ignore_unmapped"]
        for m in ("unet", "segformer", "fusion"):
            s = v[m]
            print(f"  {m:9s} mean {s['mean_dice']:.4f}  binary {s['mean_binary_dice']:.4f} +/- {s['std_binary_dice']:.4f}"
                  f"  pooled binary {s['pooled']['pooled_binary_dice']:.4f}")

    out = RESULTS_DIR / ("openurine_external_dryrun.json" if args.dry_run else "openurine_external.json")
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\nSaved", out)


if __name__ == "__main__":
    main()
