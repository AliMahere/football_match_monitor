"""Turn a CVAT COCO export into an evaluation set and a train/test split.

Only the frames that were actually reviewed are kept: a CVAT export also carries
the unreviewed frames, still holding the model's original predictions, and those
are not ground truth.

Roles are collapsed onto the two classes every detector under comparison can
produce, so goalkeeper and referee become player.

Example:
    python scripts/prepare_labels.py \
        --cvat "../inference_out/cvat_left_3zvbc/cvat left.json" \
        --out ../inference_out/labels_left --from-frame 1940
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EVAL_CLASSES = ["ball", "player"]
COLLAPSE = {"ball": "ball", "player": "player", "goalkeeper": "player", "referee": "player"}


def frame_index(file_name):
    return int(Path(file_name).stem.split("_")[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--from-frame",
        type=int,
        default=1940,
        help="first reviewed frame index; earlier frames are discarded",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="share of frames held out, taken from the end of the clip",
    )
    args = parser.parse_args()

    export = json.loads(args.cvat.read_text())
    source_classes = {c["id"]: c["name"] for c in export["categories"]}

    kept = [i for i in export["images"] if frame_index(i["file_name"]) >= args.from_frame]
    kept.sort(key=lambda i: frame_index(i["file_name"]))
    kept_ids = {i["id"] for i in kept}

    counts = Counter()
    by_image = {i["id"]: [] for i in kept}
    for annotation in export["annotations"]:
        if annotation["image_id"] not in kept_ids:
            continue
        name = COLLAPSE.get(source_classes[annotation["category_id"]])
        if name is None:
            continue
        counts[name] += 1
        by_image[annotation["image_id"]].append((name, annotation["bbox"]))

    # Frames are 0.44s apart, so neighbours are highly correlated. Holding out the
    # tail of the clip rather than a random sample keeps the test set honest.
    split = int(len(kept) * (1 - args.test_fraction))
    splits = {"train": kept[:split], "test": kept[split:], "all": kept}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, images in splits.items():
        annotations = []
        for image in images:
            for label, bbox in by_image[image["id"]]:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image["id"],
                        "category_id": EVAL_CLASSES.index(label) + 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
        (args.out / f"{name}.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [
                        {"id": i + 1, "name": c, "supercategory": ""}
                        for i, c in enumerate(EVAL_CLASSES)
                    ],
                },
                indent=2,
            )
        )
        print(f"{name:<6} {len(images):>3} frames, {len(annotations):>4} boxes")

    first, last = kept[0]["file_name"], kept[-1]["file_name"]
    print(f"\nkept {len(kept)} reviewed frames ({first} .. {last}) -> {args.out}")
    print("classes after collapsing roles:", dict(counts))


if __name__ == "__main__":
    main()
