"""
Step 2a - lens distortion from straight pitch lines (plumb-line calibration).

    python mapping3d/plumb.py   -> work/calib/lens_{left,right}_plumb.json + undistorted images

The touchline, goal lines, halfway line and penalty-box edges are straight on the real
pitch but visibly curved in the images. We find the division-model parameters (l1, l2,
distortion centre) that make every one of them straight again. No pitch dimensions and
no camera pose are needed for this, which is why it is done first and on its own.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from lens import Lens  # noqa: E402
from paths import WORK

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(WORK, "calib")
BAND = 50   # px: mask pixels this close to a hand polyline belong to that line


def poly_dist(P, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.einsum("nmk,mk->nm", P[:, None] - a[None], ab) / (ab ** 2).sum(1)
    t = np.clip(t, 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.sqrt(((P[:, None] - q) ** 2).sum(-1)).min(1)


def clean(pts, tol=10.0, iters=6):
    """Keep only pixels on one smooth curve. A distorted straight line is a gentle curve,
    so a cubic along its main direction fits it; goal posts, barrier edges and the stubs
    of crossing lines stick out of that curve and get dropped."""
    for _ in range(iters):
        mu = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
        t, n = (pts - mu) @ vt[0], (pts - mu) @ vt[1]
        c = np.polyfit(t, n, 3)
        r = np.abs(n - np.polyval(c, t))
        keep = r < max(tol, 2.5 * np.median(r))
        if keep.all():
            break
        pts = pts[keep]
    return pts[r[keep] < tol] if not keep.all() else pts[r < tol]


def labelled_pixels(side, n_per_line=1500, seed=0):
    """{line name: (N,2) mask pixels} for the straight lines of one camera."""
    polys = json.load(open(os.path.join(HERE, "lines_manual.json")))[side]
    polys = {k: np.array(v, float) for k, v in polys.items() if not k.startswith("_")}
    m = cv2.imread(os.path.join(WORK, "bg", f"{side}_lines.png"), 0)
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    names = list(polys)
    D = np.stack([poly_dist(P, polys[k]) for k in names], 1)
    best = D.argmin(1)
    srt = np.sort(D, 1)
    # inside the band, and clearly nearer this line than any other (drops junctions)
    ok = (srt[:, 0] < BAND) & (srt[:, 1] > srt[:, 0] + BAND)
    rng = np.random.default_rng(seed)
    out = {}
    for i, k in enumerate(names):
        pts = clean(P[ok & (best == i)])
        if len(pts) > n_per_line:
            pts = pts[rng.choice(len(pts), n_per_line, replace=False)]
        out[k] = pts
    return out


def straightness(q, groups):
    lens = Lens(q[0], q[1])          # distortion centre fixed at the image centre
    res = []
    for pts in groups.values():
        u = lens.undistort(pts)
        mu = u.mean(0)
        _, _, vt = np.linalg.svd(u - mu, full_matrices=False)
        # distance to the best straight line, converted back to ORIGINAL image pixels
        # (divide by the local magnification of the undistortion). Measured this way
        # the fit can't cheat by stretching the image until every line looks short.
        r2 = (((pts - lens.c) / 1920.0) ** 2).sum(1)
        mag = 1.0 / (1 + lens.l1 * r2 + lens.l2 * r2 ** 2)
        res.append((u - mu) @ vt[1] / mag)
    # the model must stay well-behaved (no fold-over) out to the image corners
    r2 = np.linspace(0, 1.2, 50) ** 2
    den = 1 + q[0] * r2 + q[1] * r2 ** 2
    res.append(np.maximum(0, 0.35 - den) * 1e4)
    return np.concatenate(res)


def rms(lens, groups):
    """RMS distance (original px) of line pixels from their straight line."""
    return float(np.sqrt(np.mean(straightness([lens.l1, lens.l2], groups)[:-50] ** 2)))


def fit(side):
    groups = labelled_pixels(side)
    r = least_squares(straightness, [-0.1, 0.0], args=(groups,), loss="soft_l1",
                      f_scale=3.0, bounds=([-0.8, -0.5], [0.3, 0.5]))
    lens = Lens(r.x[0], r.x[1])
    return lens, groups, rms(Lens(), groups), rms(lens, groups)


def undistort_image(img, lens, scale=1.0):
    """Remap a whole image to its ideal-pinhole version (same size)."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xu = np.c_[xs.ravel(), ys.ravel()]
    # centre the undistorted image and shrink it by `scale` to fit the stretched corners
    xu = lens.c + (xu - np.array([w / 2, h / 2])) / scale
    xd = lens.distort(xu).astype(np.float32)
    return cv2.remap(img, xd[:, 0].reshape(h, w), xd[:, 1].reshape(h, w), cv2.INTER_LINEAR)


def main():
    os.makedirs(OUT, exist_ok=True)
    for side in ("left", "right"):
        lens, groups, before, after = fit(side)
        info = {"model": "division", "l1": lens.l1, "l2": lens.l2,
                "centre": lens.c.tolist(), "S": 1920.0,
                "line_rms_px_before": before, "line_rms_px_after": after,
                "lines_used": {k: int(len(v)) for k, v in groups.items()}}
        json.dump(info, open(os.path.join(OUT, f"lens_{side}_plumb.json"), "w"), indent=2)
        print(side, json.dumps(info))
        img = cv2.imread(os.path.join(WORK, "bg", f"{side}_bg.png"))
        und = undistort_image(img, lens, scale=0.55)
        cv2.imwrite(os.path.join(OUT, f"{side}_undistorted.jpg"), und)


if __name__ == "__main__":
    main()
