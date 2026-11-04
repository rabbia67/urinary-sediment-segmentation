"""Fill VOC boxes into SegmentationClass masks beside the downloaded images.

The Drive folder already uses JPEGImages, Annotations, and ImageSets/Main.
Masks are written to SegmentationClass with the same file stem. Images are
left where they are. Larger boxes are painted first, so a smaller box keeps
the pixel where two objects overlap.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "urinary-sediment"

CLASS_TO_ID = {
    "cast": 1,
    "cryst": 2,
    "epith": 3,
    "epithn": 4,
    "eryth": 5,
    "leuko": 6,
    "mycete": 7,
}
SPLITS = ("train", "val", "test")


def find_dir(name: str) -> Path:
    matches = [path for path in RAW_ROOT.rglob(name) if path.is_dir()]
    if not matches:
        raise SystemExit(f"Missing {name} under {RAW_ROOT}. The Drive download may still be running.")
    return matches[0]


def read_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stem = Path(line.strip()).stem
        if stem:
            ids.append(stem)
    return ids


def find_image(image_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".tif"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for {stem} in {image_dir}")


def boxes_to_mask(xml_path: Path, height: int, width: int) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    painted = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        class_id = CLASS_TO_ID.get(name)
        if class_id is None:
            continue
        box = obj.find("bndbox")
        x1 = int(float(box.findtext("xmin")))
        y1 = int(float(box.findtext("ymin")))
        x2 = int(float(box.findtext("xmax"))) + 1
        y2 = int(float(box.findtext("ymax"))) + 1
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        painted.append(((x2 - x1) * (y2 - y1), class_id, y1, y2, x1, x2))
    mask = np.zeros((height, width), dtype=np.uint8)
    for _area, class_id, y1, y2, x1, x2 in sorted(painted, reverse=True):
        mask[y1:y2, x1:x2] = class_id
    return mask


def main() -> None:
    annotations = find_dir("Annotations")
    images = find_dir("JPEGImages")
    split_dir = find_dir("Main")
    mask_out = RAW_ROOT / "SegmentationClass"
    mask_out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        ids = read_ids(split_dir / f"{split}.txt")
        for stem in ids:
            src = find_image(images, stem)
            xml_path = annotations / f"{stem}.xml"
            if not xml_path.exists():
                raise FileNotFoundError(xml_path)
            with Image.open(src) as image:
                width, height = image.size
            mask = boxes_to_mask(xml_path, height, width)
            Image.fromarray(mask).save(mask_out / f"{stem}.png")
        counts[split] = len(ids)
        print(f"{split}: {len(ids)} masks")
    print(counts)
    print(f"Wrote {mask_out}")


if __name__ == "__main__":
    main()
