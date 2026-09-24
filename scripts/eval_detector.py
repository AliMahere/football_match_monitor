"""Score a detector against reviewed ground truth and append the run to a log.

Predictions are collapsed onto the same two classes as the labels, so a model
that reports goalkeeper or referee is credited with a player. YOLO26 only offers
COCO person and sports ball, which is what the two-class comparison is built
around.

Example:
    python scripts/eval_detector.py --labels ../inference_out/labels_left/all.json \
        --frames ../inference_out/cvat_left_3zvbc/frames \
        --model roboflow-players --log ../inference_out/eval_3zvbc.log
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.metrics import F1Score, MeanAveragePrecision, Precision, Recall

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_frames import build_detector, sliced  # noqa: E402

EVAL_CLASSES = ["ball", "player"]
COLLAPSE = {"ball": "ball", "player": "player", "goalkeeper": "player", "referee": "player"}


def to_detections(xyxy, scores, labels):
    keep = [i for i, l in enumerate(labels) if COLLAPSE.get(l) in EVAL_CLASSES]
    if not keep:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32)[keep].reshape(-1, 4),
        confidence=np.asarray(scores, dtype=np.float32)[keep],
        class_id=np.array(
            [EVAL_CLASSES.index(COLLAPSE[labels[i]]) for i in keep], dtype=int
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--slice", action="store_true")
    parser.add_argument("--slice-wh", type=int, nargs=2, default=(1280, 1280))
    parser.add_argument("--overlap-wh", type=int, nargs=2, default=(256, 256))
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    names = {c["id"]: c["name"] for c in labels["categories"]}
    truth_by_image = {i["id"]: [] for i in labels["images"]}
    for annotation in labels["annotations"]:
        x, y, w, h = annotation["bbox"]
        truth_by_image[annotation["image_id"]].append(
            (EVAL_CLASSES.index(names[annotation["category_id"]]), [x, y, x + w, y + h])
        )

    detector = build_detector(args.model, args.imgsz, args.conf)

    predictions, targets = [], []
    predicted_counts, truth_counts = Counter(), Counter()
    elapsed = 0.0

    for image in labels["images"]:
        frame = cv2.imread(str(args.frames / image["file_name"]))
        if frame is None:
            raise SystemExit(f"missing frame: {args.frames / image['file_name']}")

        started = time.perf_counter()
        if args.slice:
            xyxy, scores, detected = sliced(
                detector, frame, tuple(args.slice_wh), tuple(args.overlap_wh)
            )
        else:
            xyxy, scores, detected = detector(frame)
        elapsed += time.perf_counter() - started

        prediction = to_detections(xyxy, scores, detected)
        predictions.append(prediction)
        for class_id in prediction.class_id:
            predicted_counts[EVAL_CLASSES[class_id]] += 1

        rows = truth_by_image[image["id"]]
        if rows:
            target = sv.Detections(
                xyxy=np.array([b for _, b in rows], dtype=np.float32),
                class_id=np.array([c for c, _ in rows], dtype=int),
            )
            for class_id, _ in rows:
                truth_counts[EVAL_CLASSES[class_id]] += 1
        else:
            target = sv.Detections.empty()
        targets.append(target)

    scores = {
        "mAP": MeanAveragePrecision().update(predictions, targets).compute(),
        "F1": F1Score().update(predictions, targets).compute(),
        "P": Precision().update(predictions, targets).compute(),
        "R": Recall().update(predictions, targets).compute(),
    }

    lines = [
        "=" * 64,
        f"{datetime.now():%Y-%m-%d %H:%M}  model={args.model}",
        f"labels={args.labels}  frames={len(labels['images'])}",
        f"imgsz={args.imgsz} conf={args.conf} slice={args.slice}"
        + (f" slice_wh={tuple(args.slice_wh)}" if args.slice else ""),
        f"ground truth: {dict(truth_counts)}",
        f"predicted:    {dict(predicted_counts)}",
        f"{elapsed / max(1, len(labels['images'])):.2f}s per frame",
        "",
        f"mAP@50    {scores['mAP'].map50:.4f}",
        f"mAP@50-95 {scores['mAP'].map50_95:.4f}",
        f"F1@50     {scores['F1'].f1_50:.4f}",
        f"P@50      {scores['P'].precision_at_50:.4f}",
        f"R@50      {scores['R'].recall_at_50:.4f}",
        "",
        "per class (AP@50):",
    ]
    for class_id, ap in zip(scores["mAP"].matched_classes, scores["mAP"].ap_per_class):
        lines.append(f"  {EVAL_CLASSES[class_id]:<8} {ap[0]:.4f}")

    report = "\n".join(lines)
    print(report)
    with args.log.open("a") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    main()
