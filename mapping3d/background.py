"""
Step 2a - player-free background and white-line mask for each camera.

    python mapping3d/background.py   -> work/bg/{left,right}_bg.png, work/bg/{left,right}_lines.png

The median of 45 frames spread over the whole clip: players move, so they vanish, and
the painted lines stay. Lines are then found as thin bright structures: a top-hat filter
(brighter than the surroundings within ~50 px), a threshold, and removal of small blobs
(grass speckle). Stray pixels in trees and fences are left in on purpose - the
calibration only uses pixels close to a projected pitch marking.
"""
import os

import cv2
import numpy as np
from paths import FOOTAGE, WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "bg")


def background(side, n=45):
    cap = cv2.VideoCapture(os.path.join(FOOTAGE, f"{side.capitalize()} camera (stereo pair).mp4"))
    frames = []
    for i in np.linspace(0, 13400, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return np.median(np.array(frames), axis=0).astype(np.uint8)


def line_mask(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    m = (th > 32).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    extent = np.maximum(st[:, 2], st[:, 3])
    keep = np.flatnonzero((extent > 120) & (st[:, 4] > 300))
    keep = keep[keep > 0]
    return np.isin(lab, keep).astype(np.uint8) * 255


def main():
    os.makedirs(OUT, exist_ok=True)
    for side in ("left", "right"):
        bg = background(side)
        cv2.imwrite(os.path.join(OUT, f"{side}_bg.png"), bg)
        cv2.imwrite(os.path.join(OUT, f"{side}_lines.png"), line_mask(bg))
        print(side, "done")


if __name__ == "__main__":
    main()
