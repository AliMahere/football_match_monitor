"""
Step 1 - measure the time offset between the LEFT and RIGHT videos.

    python mapping3d/sync.py      -> work/sync/sync.json, work/sync/lag_curve.png

Why not simpler methods (all tried, see README):
  * audio: the stereo-pair files have no audio track.
  * timestamps: both files are clips cut from longer recordings. Each starts with a few
    "pre-roll" frames with negative timestamps (left 175, right 54) that players and
    OpenCV skip. The frame counts in the header include them, which is why they differ
    by 121, but that says nothing about when each camera actually started.
  * global motion / brightness signals: dominated by video-compression noise; they gave
    different answers on different parts of the clip.

What works: geometry. With both cameras calibrated to one pitch, a player's feet map to
the same (x, y) in metres from either camera - but only if both frames show the same
moment. So for each candidate lag we take the players standing where BOTH cameras can
see, map their feet to the pitch from each camera, and measure how far apart the two
sets of positions are. The right lag makes them coincide.

Convention: right_frame = left_frame + lag.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geometry import H_IMG, W_IMG, load_pair  # noqa: E402
from paths import WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "sync")
FPS = 45.0
CAP = 2.0   # m: a player with no partner within this counts as this far off


def load(side, tag=""):
    d = np.load(os.path.join(OUT, f"players_{side}{tag}.npz"))
    frames, counts, boxes = d["frames"], d["counts"], d["boxes"]
    split = np.split(boxes, np.cumsum(counts)[:-1])
    return dict(zip(frames.tolist(), split))


def feet_in_overlap(boxes, cam, other):
    """Ground position (x, y) of each player that both cameras can see."""
    if len(boxes) == 0:
        return np.zeros((0, 2))
    x1, y1, x2, y2 = boxes.T
    # a box cut off by the image border has no visible feet
    ok = (y2 < H_IMG - 10) & (x1 > 10) & (x2 < W_IMG - 10)
    feet = np.c_[(x1 + x2) / 2, y2][ok]
    if len(feet) == 0:
        return np.zeros((0, 2))
    P = cam.to_plane(feet, 0.0)
    good = ~np.isnan(P).any(1)
    P = P[good]
    both = cam.sees(P, 60) & other.sees(P, 60)
    return P[both][:, :2]


def chamfer(A, B):
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    return np.minimum(D.min(1), CAP).mean() / 2 + np.minimum(D.min(0), CAP).mean() / 2


def measure(camL, camR, tag=""):
    """Lag curve for one stretch of detections."""
    L, R = load("left", tag), load("right", tag)
    FL = {t: feet_in_overlap(b, camL, camR) for t, b in L.items()}
    FR = {t: feet_in_overlap(b, camR, camL) for t, b in R.items()}
    lags = np.arange(-900, 901)
    score, used = np.full(len(lags), np.nan), np.zeros(len(lags), int)
    for i, lag in enumerate(lags):
        vals = []
        for t, A in FL.items():
            B = FR.get(t + lag)
            if B is not None and len(A) and len(B):
                vals.append(chamfer(A, B))
        if len(vals) > 200:
            score[i], used[i] = np.mean(vals), len(vals)
    i = int(np.nanargmin(score))
    best = int(lags[i])
    far = np.abs(lags - best) > 45          # how sharp is the minimum?
    y0, y1, y2 = score[i - 1], score[i], score[i + 1]
    sub = best + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)   # parabola: sub-frame estimate
    return {"left_frames": [min(L), max(L)], "lag_frames": best, "lag_subframe": float(sub),
            "mean_position_mismatch_m_at_lag": float(np.nanmin(score)),
            "median_mismatch_m_elsewhere": float(np.nanmedian(score[far])),
            "frames_compared": int(used[i]),
            "players_in_overlap_per_frame": {
                "left": float(np.mean([len(v) for v in FL.values()])),
                "right": float(np.mean([len(v) for v in FR.values()]))}}, lags, score


def main():
    camL, camR = load_pair()
    early, lags, score = measure(camL, camR)
    res = {"lag_frames": early["lag_frames"], "lag_seconds": early["lag_frames"] / FPS,
           "convention": "right_frame = left_frame + lag", "early": early}
    if os.path.exists(os.path.join(OUT, "players_left_late.npz")):
        late, lags2, score2 = measure(camL, camR, "_late")
        res["late"] = late
        span = (late["left_frames"][0] - early["left_frames"][0]) / FPS
        res["drift_frames_per_minute"] = (late["lag_subframe"] - early["lag_subframe"]) / span * 60
    json.dump(res, open(os.path.join(OUT, "sync.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "lag_curve.npz"), lags=lags, score=score)
    print(json.dumps(res, indent=1))
    best = res["lag_frames"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, rng in zip(ax, (None, 60)):
        sel = slice(None) if rng is None else (np.abs(lags - best) <= rng)
        a.plot(lags[sel], score[sel], lw=1.2, label="frames 0-4000")
        if "late" in res:
            a.plot(lags2[sel], score2[sel], lw=1.2, label="frames 9000-11000")
        a.axvline(best, color="C3", ls="--", lw=1)
        a.set_xlabel("lag (frames): right = left + lag")
        a.set_ylabel("mean foot-position mismatch (m)")
    ax[0].set_title("Players in the overlap, mapped to the pitch from each camera")
    ax[1].set_title(f"zoom: minimum at {best} frames ({best / FPS:+.2f} s)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lag_curve.png"), dpi=120)


if __name__ == "__main__":
    main()
