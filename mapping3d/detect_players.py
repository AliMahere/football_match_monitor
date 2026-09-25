"""
Player boxes for every frame of a stretch of both videos (for measuring the time offset).

    python mapping3d/detect_players.py --frames 0 4000   -> work/sync/players_{left,right}.npz

Same detector the tracker uses for players: stock YOLO26x on the whole frame (players are
~400 px tall at 4K, so no tiling is needed). No tracking here - sync only needs where the
people are, not who they are.
"""
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO
from paths import FOOTAGE, WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "yolo26x.pt")


def run(side, start, stop, model, batch=8):
    cap = cv2.VideoCapture(os.path.join(FOOTAGE, f"{side.capitalize()} camera (stereo pair).mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, boxes, buf, idx = [], [], [], start
    while idx < stop:
        ok, img = cap.read()
        if not ok:
            break
        buf.append(img)
        frames.append(idx)
        idx += 1
        if len(buf) == batch or idx == stop:
            for r in model.predict(buf, imgsz=1280, conf=0.3, classes=[0], verbose=False):
                boxes.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
            buf = []
    if buf:
        for r in model.predict(buf, imgsz=1280, conf=0.3, classes=[0], verbose=False):
            boxes.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
    return np.array(frames), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs=2, type=int, default=[0, 4000])
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args()
    out = os.path.join(WORK, "sync")
    os.makedirs(out, exist_ok=True)
    model = YOLO(WEIGHTS)
    for side in ("left", "right"):
        frames, boxes = run(side, *a.frames, model)
        n = np.array([len(b) for b in boxes])
        flat = np.concatenate([b for b in boxes if len(b)]) if n.sum() else np.zeros((0, 4))
        np.savez(os.path.join(out, f"players_{side}{a.tag}.npz"), frames=frames, counts=n, boxes=flat)
        print(side, len(frames), "frames,", n.mean().round(2), "people per frame")


if __name__ == "__main__":
    main()
