"""Sample frames from a match video, detect players and the ball, and write
annotated previews plus COCO pre-annotations for review in CVAT.

Example:
    python detection/infer_frames.py \
        --video "../raw_footage/Left camera (stereo pair).mp4" \
        --model rfdetr-soccernet \
        --out ../inference_out/left_rfdetr \
        --frames 12 --start 1000 --stride 1000
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Every model is mapped onto this label set so a CVAT task holds the same
# classes regardless of which detector pre-annotated it.
CLASSES = ["ball", "player", "goalkeeper", "referee"]
COLORS = {
    "ball": (0, 0, 255),
    "player": (0, 255, 0),
    "goalkeeper": (255, 255, 0),
    "referee": (0, 255, 255),
}


class UltralyticsDetector:
    def __init__(self, weights, label_map, imgsz, conf):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.label_map = label_map
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, image):
        result = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        names = [self.model.names[int(c)] for c in boxes.cls.cpu().numpy()]

        keep = [i for i, n in enumerate(names) if n in self.label_map]
        labels = [self.label_map[names[i]] for i in keep]
        return xyxy[keep], scores[keep], labels


class RFDETRDetector:
    def __init__(self, weights, label_map, resolution, conf):
        import rfdetr

        # The published SoccerNet checkpoint predates the patch-16 rewrite, so the
        # deprecated patch-14 class is the one that actually accepts it, and its
        # input resolution has to be a multiple of patch_size * num_windows.
        block = 56
        resolution = max(block, round(resolution / block) * block)

        self.model = None
        errors = []
        for variant in ("RFDETRLargeDeprecated", "RFDETRBase", "RFDETRLarge"):
            if not hasattr(rfdetr, variant):
                continue
            try:
                self.model = getattr(rfdetr, variant)(
                    pretrain_weights=str(weights), resolution=resolution
                )
                print(f"loaded {weights.name} as {variant}")
                break
            except Exception as exc:  # checkpoint only fits one variant
                errors.append(f"{variant}: {exc}")
        if self.model is None:
            raise RuntimeError("could not load RF-DETR checkpoint\n" + "\n".join(errors))

        self.label_map = label_map
        self.conf = conf

    def __call__(self, image):
        from PIL import Image

        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = self.model.predict(rgb, threshold=self.conf)
        if len(detections) == 0:
            return np.zeros((0, 4)), np.zeros(0), []

        names = [self.label_map.get(int(c)) for c in detections.class_id]
        keep = [i for i, n in enumerate(names) if n is not None]
        return (
            detections.xyxy[keep],
            detections.confidence[keep],
            [names[i] for i in keep],
        )


class RoboflowDetector:
    """Roboflow-hosted weights executed locally. The frames are never uploaded."""

    def __init__(self, model_id, label_map, conf):
        import os

        from inference import get_model

        api_key = os.environ.get("ROBOFLOW_API_PRIVATE_KEY")
        if not api_key:
            raise SystemExit("set ROBOFLOW_API_PRIVATE_KEY (see .env) to use this model")

        self.model = get_model(model_id=model_id, api_key=api_key)
        self.label_map = label_map
        self.conf = conf

    def __call__(self, image):
        predictions = self.model.infer(image, confidence=self.conf)[0].predictions
        xyxy, scores, labels = [], [], []
        for p in predictions:
            label = self.label_map.get(p.class_name)
            if label is None:
                continue
            # Roboflow reports centre and size rather than corners.
            xyxy.append(
                [
                    p.x - p.width / 2,
                    p.y - p.height / 2,
                    p.x + p.width / 2,
                    p.y + p.height / 2,
                ]
            )
            scores.append(p.confidence)
            labels.append(label)
        return (
            np.array(xyxy, dtype=float).reshape(-1, 4),
            np.array(scores, dtype=float),
            labels,
        )


def build_detector(name, imgsz, conf):
    """Return a callable mapping a BGR frame to (xyxy, scores, labels)."""
    from huggingface_hub import hf_hub_download

    if name.endswith(".pt") and Path(name).exists():
        # A checkpoint we fine-tuned ourselves already uses the shared class names.
        return UltralyticsDetector(name, {c: c for c in CLASSES}, imgsz, conf)

    if name == "yolo26":
        # COCO weights: person and sports ball are the only useful classes.
        return UltralyticsDetector(
            "yolo26x.pt", {"person": "player", "sports ball": "ball"}, imgsz, conf
        )

    if name == "yolo11m-player":
        weights = hf_hub_download(
            "martinjolif/yolo-football-player-detection",
            "yolo-football-player-detection.pt",
        )
        return UltralyticsDetector(
            weights, {c: c for c in CLASSES}, imgsz, conf
        )

    if name == "yolo11n-ball":
        weights = hf_hub_download(
            "martinjolif/yolo-football-ball-detection",
            "yolo-football-ball-detection.pt",
        )
        return UltralyticsDetector(weights, {"ball": "ball"}, imgsz, conf)

    if name == "roboflow-ball":
        return RoboflowDetector(
            "football-ball-detection-rejhg/4", {"ball": "ball"}, conf
        )

    if name == "roboflow-players":
        return RoboflowDetector(
            "football-players-detection-3zvbc/20", {c: c for c in CLASSES}, conf
        )

    if name == "rfdetr-soccernet":
        weights = hf_hub_download(
            "julianzu9612/RFDETR-Soccernet", "weights/checkpoint_best_regular.pth"
        )
        # Class order is given by the model card: 0 ball, 1 player, 2 referee, 3 goalkeeper.
        label_map = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}
        return RFDETRDetector(Path(weights), label_map, imgsz, conf)

    raise ValueError(f"unknown model: {name}")


def sliced(detector, image, slice_wh, overlap_wh):
    """Run the detector over overlapping tiles so small objects survive at native
    resolution instead of being downscaled away. Overlap is in pixels and must
    comfortably exceed the largest object, or players will be cut between tiles."""
    import supervision as sv

    def callback(tile):
        xyxy, scores, labels = detector(tile)
        class_id = np.array([CLASSES.index(l) for l in labels], dtype=int)
        return sv.Detections(
            xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(scores, dtype=np.float32),
            class_id=class_id,
        )

    slicer = sv.InferenceSlicer(
        callback=callback, slice_wh=slice_wh, overlap_wh=overlap_wh
    )
    merged = slicer(image)
    labels = [CLASSES[i] for i in merged.class_id]
    return merged.xyxy, merged.confidence, labels


def draw(image, xyxy, scores, labels):
    preview = image.copy()
    thickness = max(2, image.shape[1] // 960)
    for (x1, y1, x2, y2), score, label in zip(xyxy, scores, labels):
        color = COLORS[label]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(preview, p1, p2, color, thickness)
        cv2.putText(
            preview,
            f"{label} {score:.2f}",
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness * 0.35,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="a name from the registry in build_detector, or a path to a .pt checkpoint",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--stride", type=int, default=1000, help="frames between samples")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--slice",
        action="store_true",
        help="tile the frame at native resolution instead of downscaling it",
    )
    parser.add_argument("--slice-wh", type=int, nargs=2, default=(1280, 1280))
    parser.add_argument("--overlap-wh", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--preview-width", type=int, default=1920)
    args = parser.parse_args()

    frames_dir = args.out / "frames"
    preview_dir = args.out / "preview"
    frames_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(args.model, args.imgsz, args.conf)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    images, annotations = [], []
    elapsed = 0.0

    for n in range(args.frames):
        index = args.start + n * args.stride
        if index >= total:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            print(f"could not read frame {index}")
            continue

        started = time.perf_counter()
        if args.slice:
            xyxy, scores, labels = sliced(
                detector, frame, tuple(args.slice_wh), tuple(args.overlap_wh)
            )
        else:
            xyxy, scores, labels = detector(frame)
        elapsed += time.perf_counter() - started

        name = f"frame_{index:06d}.jpg"
        height, width = frame.shape[:2]
        cv2.imwrite(str(frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        preview = draw(frame, xyxy, scores, labels)
        scale = args.preview_width / width
        cv2.imwrite(
            str(preview_dir / name),
            cv2.resize(preview, (args.preview_width, int(height * scale))),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

        image_id = len(images) + 1
        images.append(
            {"id": image_id, "file_name": name, "width": width, "height": height}
        )
        for (x1, y1, x2, y2), score, label in zip(xyxy, scores, labels):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASSES.index(label) + 1,
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2 - x1), 2),
                        round(float(y2 - y1), 2),
                    ],
                    "area": round(float((x2 - x1) * (y2 - y1)), 2),
                    "iscrowd": 0,
                    "score": round(float(score), 4),
                }
            )

        counts = {c: labels.count(c) for c in CLASSES if labels.count(c)}
        print(f"{name}: {counts or 'nothing detected'}")

    capture.release()

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(CLASSES)
        ],
    }
    (args.out / "annotations_coco.json").write_text(json.dumps(coco, indent=2))
    (args.out / "run.json").write_text(json.dumps(vars(args), indent=2, default=str))

    per_frame = elapsed / len(images) if images else 0
    print(
        f"\n{len(images)} frames, {len(annotations)} detections, "
        f"{per_frame:.2f}s per frame -> {args.out}"
    )


if __name__ == "__main__":
    main()
