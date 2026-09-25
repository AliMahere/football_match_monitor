"""Merge the reviewed COCO splits from both cameras into one YOLO dataset.

Frames from the two cameras share file names, so each is prefixed with its
camera before being copied into the dataset.

Example:
    python detection/build_dataset.py \
        --camera left ../inference_out/labels_left ../inference_out/cvat_left_3zvbc/frames \
        --camera right ../inference_out/labels_right ../inference_out/cvat_right_3zvbc/frames \
        --out ../datasets/ours
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

CLASSES = ["ball", "player"]


def write_split(split, cameras, out):
    images_dir = out / split / "images"
    labels_dir = out / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    frames = 0

    for camera, labels_root, frames_root in cameras:
        coco = json.loads((labels_root / f"{split}.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = {i["id"]: [] for i in coco["images"]}
        for annotation in coco["annotations"]:
            by_image[annotation["image_id"]].append(annotation)

        for image in coco["images"]:
            source = frames_root / image["file_name"]
            if not source.exists():
                raise SystemExit(f"missing frame: {source}")

            stem = f"{camera}_{Path(image['file_name']).stem}"
            shutil.copy2(source, images_dir / f"{stem}.jpg")
            frames += 1

            width, height = image["width"], image["height"]
            rows = []
            for annotation in by_image[image["id"]]:
                name = names[annotation["category_id"]]
                x, y, w, h = annotation["bbox"]
                rows.append(
                    f"{CLASSES.index(name)} "
                    f"{(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                    f"{w / width:.6f} {h / height:.6f}"
                )
                counts[name] += 1
            (labels_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n")

    print(f"{split:<6} {frames:>3} images, {sum(counts.values()):>4} boxes  {dict(counts)}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "LABELS_DIR", "FRAMES_DIR"),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cameras = [(n, Path(labels), Path(frames)) for n, labels, frames in args.camera]
    for split in ("train", "test"):
        write_split(split, cameras, args.out)

    # Ultralytics wants a val key; the held-out tail serves as both. The path must be
    # absolute, since a relative one resolves against the working directory.
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
