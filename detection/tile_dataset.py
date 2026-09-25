"""Cut a YOLO dataset of 4K frames into native-resolution tiles.

Training on whole 4K frames means resizing them to the network input, which
shrinks the ball from about 34 pixels to about 11 and effectively removes it.
Tiling keeps every object at its original scale, and matches how the detector is
run at inference with --slice.

Tiles holding no object are mostly dropped, but a share is kept on purpose: the
pitch markings that get mistaken for the ball only appear in empty tiles, and the
model needs to see them labelled as background.

Example:
    python detection/tile_dataset.py --src ../datasets/ours --out ../datasets/ours_tiled
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2

CLASSES = ["ball", "player"]


def tile_origins(total, tile, overlap):
    step = tile - overlap
    origins = list(range(0, max(1, total - tile + 1), step))
    if origins[-1] + tile < total:
        origins.append(total - tile)
    return origins


def process(split, src, out, tile, overlap, keep_empty, min_visible, rng):
    images_dir = out / split / "images"
    labels_dir = out / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    written = empty_kept = 0

    for image_path in sorted((src / split / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        boxes = []
        label_path = src / split / "labels" / f"{image_path.stem}.txt"
        for line in label_path.read_text().split("\n"):
            if not line.strip():
                continue
            class_id, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            boxes.append((int(class_id), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

        for top in tile_origins(height, tile, overlap):
            for left in tile_origins(width, tile, overlap):
                rows = []
                for class_id, x1, y1, x2, y2 in boxes:
                    # Clip the box to the tile and keep it only if most of it survives.
                    cx1, cy1 = max(x1, left), max(y1, top)
                    cx2, cy2 = min(x2, left + tile), min(y2, top + tile)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    if (cx2 - cx1) * (cy2 - cy1) < min_visible * (x2 - x1) * (y2 - y1):
                        continue
                    rows.append(
                        f"{class_id} "
                        f"{((cx1 + cx2) / 2 - left) / tile:.6f} "
                        f"{((cy1 + cy2) / 2 - top) / tile:.6f} "
                        f"{(cx2 - cx1) / tile:.6f} {(cy2 - cy1) / tile:.6f}"
                    )
                    counts[CLASSES[class_id]] += 1

                if not rows:
                    if rng.random() > keep_empty:
                        continue
                    empty_kept += 1

                stem = f"{image_path.stem}_{left}_{top}"
                cv2.imwrite(
                    str(images_dir / f"{stem}.jpg"),
                    image[top : top + tile, left : left + tile],
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                (labels_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                written += 1

    print(
        f"{split:<6} {written:>4} tiles ({empty_kept} empty kept), "
        f"{sum(counts.values()):>4} boxes  {dict(counts)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument(
        "--keep-empty",
        type=float,
        default=0.15,
        help="share of object-free tiles to keep as background examples",
    )
    parser.add_argument(
        "--min-visible",
        type=float,
        default=0.4,
        help="drop a box cut by a tile edge unless this much of its area remains",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split in ("train", "test"):
        process(
            split, args.src, args.out, args.tile, args.overlap,
            args.keep_empty, args.min_visible, rng,
        )

    # Ultralytics resolves a relative path against the working directory, not the
    # file, so it has to be absolute.
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
