"""
Joint calibration of the LEFT and RIGHT cameras from the pitch markings alone.

    python mapping3d/calibrate.py            -> work/calib/calibration.json + overlay images

Runs AFTER mapping3d/plumb.py, which already removed the lens distortion. Everything here
works in undistorted (ideal pinhole) pixel coordinates.

Unknowns (all estimated together, one least-squares problem):
    per camera : focal length f, rotation (3), position (3)
    shared     : pitch length, width, penalty box depth/width, penalty spot distance,
                 centre-circle radius, "D" arc radius/centre, and how far each goal
                 frame stands behind its goal line

Observations:
    1. ~13 hand-read keypoints per camera (mapping3d/keypoints_manual.json) - only to get
       started; they are approximate.
    2. Thousands of white line pixels from the player-free background image. Each pixel
       is assigned to the nearest projected marking (ICP style) and its distance to that
       marking is minimised - including the centre circle and the D arcs, which the
       plumb-line step could not use.
    3. The goal frames: posts 3 m apart, 2 m tall. This is the ONLY metric assumption,
       and it fixes the scale of everything (a camera can't tell a big pitch far away from
       a small one close up).

Because both cameras are fitted against ONE pitch, their poses come out in the same
world frame. That is the stereo calibration: the baseline between them falls out of it.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
import pitch  # noqa: E402
from lens import Lens  # noqa: E402
from paths import WORK

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(WORK, "calib")
W_IMG, H_IMG = 3840, 2160
CX, CY = W_IMG / 2, H_IMG / 2
SIDES = ("left", "right")
N_DIM = len(pitch.DIM_NAMES)
N_CAM = 7  # f, rvec(3), tvec(3)   (lens distortion comes from plumb.py)


# ----------------------------------------------------------------------------- camera
def K_of(f):
    return np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.0]])


def split(p):
    dims = p[:N_DIM]
    e = p[N_DIM:N_DIM + 2]
    cams = {}
    for i, s in enumerate(SIDES):
        c = p[N_DIM + 2 + i * N_CAM: N_DIM + 2 + (i + 1) * N_CAM]
        cams[s] = {"f": c[0], "rvec": c[1:4], "tvec": c[4:7]}
    return dims, e, cams


def project(cam, X):
    """World points (N,3) -> UNDISTORTED pixels (N,2); NaN behind the camera."""
    if len(X) == 0:
        return np.zeros((0, 2))
    R, _ = cv2.Rodrigues(cam["rvec"])
    Xc = X @ R.T + cam["tvec"]
    uv = cam["f"] * Xc[:, :2] / np.maximum(Xc[:, 2:3], 1e-6) + np.array([CX, CY])
    uv[Xc[:, 2] < 0.5] = np.nan
    # far outside the real field of view: not observable, don't let it attract pixels
    uv[(np.abs(uv - [CX, CY]) > [6000, 4000]).any(1)] = np.nan
    return uv


def seg_dist(P, poly):
    """Distance from each point in P (N,2) to the polyline poly (M,2), NaN-safe."""
    good = ~np.isnan(poly).any(1)
    a, b = poly[:-1], poly[1:]
    keep = good[:-1] & good[1:]
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return np.full(len(P), 1e4)
    ab = b - a
    t = np.einsum("nmk,mk->nm", P[:, None, :] - a[None], ab) / np.maximum((ab**2).sum(1), 1e-9)
    t = np.clip(t, 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1)


# ----------------------------------------------------------------------------- data
def load_lens(side, which="plumb"):
    """which='plumb' -> plumb.py result (the starting point); 'final' -> after stage C."""
    name = f"lens_{side}_plumb.json" if which == "plumb" else f"lens_{side}.json"
    j = json.load(open(os.path.join(OUT, name)))
    return Lens(j["l1"], j["l2"], *j["centre"])


def load_line_pixels(side, lens, n=5000, seed=0):
    """White-line pixels, undistorted. Pixels above the barrier (trees, fences) are
    dropped by the assignment gate later, since no marking projects there."""
    m = cv2.imread(os.path.join(WORK, "bg", f"{side}_lines.png"), 0)
    ys, xs = np.nonzero(m)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), min(n * 3, len(xs)), replace=False)
    return lens.undistort(np.c_[xs[idx], ys[idx]].astype(float))


def lookat(C, target):
    """rvec, tvec of a camera at C looking at target, image x to the right."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), -R @ C


# Seeds: both cameras stand behind the NEAR touchline around midfield, high up,
# the left one looking at the left half and the right one at the right half.
SEED = {"left": ([-2.0, -16.0, 7.0], [-10.0, 2.0, 0.0]),
        "right": ([2.0, -16.0, 7.0], [10.0, 2.0, 0.0])}


def init_camera(side, kp, dims):
    """Undistorted images are pinhole, so PnP works. Focal length is unknown: scan it
    and keep the one with the smallest reprojection error, seeding PnP from a
    plausible pose (behind the near touchline, high up)."""
    world = pitch.keypoints(dims)
    names = [k for k in kp if k in world]
    X = np.array([world[k] for k in names])
    x = np.array([kp[k] for k in names], float)
    rv0, tv0 = lookat(np.array(SEED[side][0]), np.array(SEED[side][1]))
    best = None
    for f in np.arange(900, 3001, 50):
        ok, rv, tv = cv2.solvePnP(X, x, K_of(f), None, rv0.copy(), tv0.copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        uv, _ = cv2.projectPoints(X, rv, tv, K_of(f), None)
        err = np.median(np.linalg.norm(uv.reshape(-1, 2) - x, axis=1))
        if ok and (best is None or err < best[0]):
            best = (err, f, rv.ravel(), tv.ravel())
    print(f"  init {side}: f={best[1]} median keypoint err {best[0]:.1f}px")
    return np.r_[best[1], best[2], best[3]]


# ----------------------------------------------------------------------------- residuals
def residuals(p, kps, lines, assign, w_line):
    dims, e, cams = split(p)
    world = pitch.keypoints(dims, e)
    curves = pitch.curves(dims)
    res = []
    for s in SIDES:
        cam = cams[s]
        names = [k for k in kps[s] if k in world]
        uv = project(cam, np.array([world[k] for k in names]))
        d = uv - np.array([kps[s][k] for k in names])
        wt = np.array([3.0 if k.startswith("post") else 1.0 for k in names])[:, None]
        d = np.nan_to_num(d, nan=200.0) * wt
        res.append(d.ravel() / 6.0)        # keypoints are good to ~6 px
        if lines is not None:
            P, lab = lines[s], assign[s]
            r = np.zeros(len(P))
            for name, poly in curves.items():
                sel = lab == name
                if sel.any():
                    r[sel] = seg_dist(P[sel], project(cam, poly))
            res.append(w_line * r / 4.0)   # line pixels are good to ~4 px (half width)
    # weak priors keep unobserved dimensions sane
    res.append((dims - pitch.DIM_INIT) / np.array([30, 30, 10, 10, 10, 5, 5, 3]) * 0.1)
    res.append(e / 1.0)
    return np.concatenate(res)


def assign_pixels(p, lines, gate):
    dims, e, cams = split(p)
    curves = pitch.curves(dims)
    out, kept = {}, {}
    for s in SIDES:
        P = lines[s]
        D = np.stack([seg_dist(P, project(cams[s], poly)) for poly in curves.values()], 1)
        best = D.argmin(1)
        ok = D[np.arange(len(P)), best] < gate
        names = np.array(list(curves))
        out[s] = names[best][ok]
        kept[s] = P[ok]
    return kept, out


def draw(p, side, path, lines=None):
    dims, e, cams = split(p)
    im = cv2.imread(os.path.join(WORK, "bg", f"{side}_bg.png"))
    lens = load_lens(side, "final")
    proj = lambda X: lens.distort(project(cams[side], X))   # back into the real image
    for poly in pitch.curves(dims).values():
        uv = proj(poly)
        for a, b in zip(uv[:-1], uv[1:]):
            if not (np.isnan(a).any() or np.isnan(b).any()):
                cv2.line(im, tuple(np.int32(a)), tuple(np.int32(b)), (0, 0, 255), 3)
    world = pitch.keypoints(dims, e)
    # goal frames as 3D boxes
    for k, X in world.items():
        uv = proj(X[None])[0]
        if not np.isnan(uv).any():
            cv2.circle(im, tuple(np.int32(uv)), 9, (0, 255, 255), 2)
    for gside, sgn in (("L", -1), ("R", 1)):
        pts = [world[f"post_{a}_{b}_{gside}"] for a, b in
               (("near", "base"), ("near", "top"), ("far", "top"), ("far", "base"))]
        uv = proj(np.array(pts))
        if not np.isnan(uv).any():
            cv2.polylines(im, [np.int32(uv)], False, (255, 0, 255), 3)
    cv2.imwrite(path, im)


def summarise(p, cost_px):
    dims, e, cams = split(p)
    out = {"world_frame": "origin centre spot; x towards right camera's goal; "
                          "y towards far touchline; z up; metres",
           "pitch": {k: float(v) for k, v in pitch.dims_dict(dims).items()},
           "goal_offset_behind_line_m": {"L": float(e[0]), "R": float(e[1])},
           "image_size": [W_IMG, H_IMG], "cameras": {}}
    centres = {}
    for s in SIDES:
        c = cams[s]
        R, _ = cv2.Rodrigues(c["rvec"])
        C = -R.T @ c["tvec"]
        centres[s] = C
        fwd = R.T @ np.array([0, 0, 1.0])
        out["cameras"][s] = {
            "K_undistorted": K_of(c["f"]).tolist(),
            "lens_division": load_lens(s, "final").to_list(),
            "rvec": c["rvec"].tolist(), "tvec": c["tvec"].tolist(),
            "centre_world_m": C.tolist(),
            "height_m": float(C[2]),
            "tilt_down_deg": float(np.degrees(np.arcsin(-fwd[2]))),
            "line_rms_px": cost_px.get(s),
        }
    out["baseline_m"] = float(np.linalg.norm(centres["left"] - centres["right"]))
    out["baseline_vector_m"] = (centres["right"] - centres["left"]).tolist()
    return out


# ----------------------------------------------------------------------------- stage C
# Full bundle adjustment: lens (l1, l2, centre) is freed too and everything is compared
# in RAW image pixels. The plumb-line lens could only use straight lines; here the centre
# circle and D arcs - which sit in the heavily bent image edges - constrain it as well.
N_LENS = 4


def split_full(q):
    p, lz = q[:-2 * N_LENS], q[-2 * N_LENS:]
    return p, {s: Lens(*lz[i * N_LENS:(i + 1) * N_LENS]) for i, s in enumerate(SIDES)}


def residuals_raw(q, kraw, lines, assign, lens0):
    p, lenses = split_full(q)
    dims, e, cams = split(p)
    world = pitch.keypoints(dims, e)
    curves = pitch.curves(dims)
    res = []
    for s in SIDES:
        cam, ls = cams[s], lenses[s]
        pr = lambda X: ls.distort(project(cam, X))
        names = list(kraw[s])
        d = pr(np.array([world[k] for k in names])) - np.array([kraw[s][k] for k in names])
        wt = np.array([3.0 if k.startswith("post") else 1.0 for k in names])[:, None]
        res.append(np.nan_to_num(d, nan=200.0).ravel() * wt.repeat(2, 1).ravel() / 6.0)
        P, lab = lines[s], assign[s]
        r = np.zeros(len(P))
        for name, poly in curves.items():
            sel = lab == name
            if sel.any():
                r[sel] = seg_dist(P[sel], pr(poly))
        res.append(r / 4.0)
        # lens centre should stay near the image centre; distortion near the plumb fit
        res.append((ls.c - [CX, CY]) / 150.0)
        res.append([(ls.l1 - lens0[s].l1) / 0.2, (ls.l2 - lens0[s].l2) / 0.2])
    res.append((dims - pitch.DIM_INIT) / np.array([30, 30, 10, 10, 10, 5, 5, 3]) * 0.1)
    res.append(e / 1.0)
    return np.concatenate(res)


def assign_raw(q, raw, gate):
    p, lenses = split_full(q)
    dims, e, cams = split(p)
    curves = pitch.curves(dims)
    names = np.array(list(curves))
    kept, out = {}, {}
    for s in SIDES:
        P = raw[s]
        D = np.stack([seg_dist(P, lenses[s].distort(project(cams[s], poly)))
                      for poly in curves.values()], 1)
        best = D.argmin(1)
        ok = D[np.arange(len(P)), best] < gate
        kept[s], out[s] = P[ok], names[best][ok]
    return kept, out


def stage_c(p, kraw, lens0):
    raw = {}
    for s in SIDES:
        m = cv2.imread(os.path.join(WORK, "bg", f"{s}_lines.png"), 0)
        ys, xs = np.nonzero(m)
        i = np.random.default_rng(2).choice(len(xs), min(15000, len(xs)), replace=False)
        raw[s] = np.c_[xs[i], ys[i]].astype(float)
    q = np.r_[p, lens0["left"].to_list(), lens0["right"].to_list()]
    for gate in (25, 15, 12):
        kept, assign = assign_raw(q, raw, gate)
        for s in SIDES:
            if len(kept[s]) > 3000:
                i = np.random.default_rng(3).choice(len(kept[s]), 3000, replace=False)
                kept[s], assign[s] = kept[s][i], assign[s][i]
        r = least_squares(residuals_raw, q, args=(kraw, kept, assign, lens0),
                          loss="soft_l1", f_scale=1.5, x_scale="jac", max_nfev=150)
        q = r.x
        print(f"stage C gate {gate}: cost {r.cost:.1f}", {s: len(kept[s]) for s in SIDES})
    p, lenses = split_full(q)
    dims, e, cams = split(p)
    curves = pitch.curves(dims)
    rms, per_marking = {}, {}
    for s in SIDES:
        d = np.zeros(len(kept[s]))
        for name, poly in curves.items():
            sel = assign[s] == name
            if sel.any():
                d[sel] = seg_dist(kept[s][sel], lenses[s].distort(project(cams[s], poly)))
                per_marking[f"{s}:{name}"] = float(np.sqrt(np.mean(d[sel] ** 2)))
        rms[s] = float(np.sqrt(np.mean(d ** 2)))
    return p, lenses, rms, per_marking


def main():
    os.makedirs(OUT, exist_ok=True)
    kraw = json.load(open(os.path.join(HERE, "keypoints_manual.json")))
    lenses = {s: load_lens(s) for s in SIDES}
    kps = {s: {k: lenses[s].undistort(np.array(v, float)) for k, v in kraw[s].items()}
           for s in SIDES}
    p = np.r_[pitch.DIM_INIT, 0.0, 0.0]
    for s in SIDES:
        p = np.r_[p, init_camera(s, kps[s], pitch.DIM_INIT)]

    # Stage A: keypoints only
    r = least_squares(residuals, p, args=(kps, None, None, 0), loss="soft_l1",
                      f_scale=2.0, x_scale="jac", max_nfev=5000)
    p = r.x
    report(p, kps, "stage A (keypoints)")

    # Stage B: ICP on every white line pixel, with a shrinking gate
    raw = {s: load_line_pixels(s, lenses[s]) for s in SIDES}
    for gate in (100, 40, 20, 14):
        kept, assign = assign_pixels(p, raw, gate)
        for s in SIDES:   # cap per camera for speed
            if len(kept[s]) > 3000:
                i = np.random.default_rng(1).choice(len(kept[s]), 3000, replace=False)
                kept[s], assign[s] = kept[s][i], assign[s][i]
        r = least_squares(residuals, p, args=(kps, kept, assign, 1.0), loss="soft_l1",
                          f_scale=1.5, x_scale="jac", max_nfev=150)
        p = r.x
        print(f"gate {gate}: cost {r.cost:.1f}", {s: len(kept[s]) for s in SIDES})
    report(p, kps, "stage B (line pixels)")
    np.save(os.path.join(OUT, "stageB.npy"), p)

    p, lenses_c, rms, per_marking = stage_c(p, kraw, lenses)
    for s in SIDES:   # the refined lens replaces the plumb-line one from here on
        j = json.load(open(os.path.join(OUT, f"lens_{s}_plumb.json")))
        j.update({"l1": lenses_c[s].l1, "l2": lenses_c[s].l2,
                  "centre": lenses_c[s].c.tolist(),
                  "refined_in": "calibrate.py stage C (full bundle adjustment)",
                  "plumb_line_only": {"l1": lenses[s].l1, "l2": lenses[s].l2,
                                      "centre": lenses[s].c.tolist()}})
        json.dump(j, open(os.path.join(OUT, f"lens_{s}.json"), "w"), indent=2)
    kps = {s: {k: lenses_c[s].undistort(np.array(v, float)) for k, v in kraw[s].items()}
           for s in SIDES}
    report(p, kps, "stage C (full bundle adjustment, raw pixels)")
    for s in SIDES:
        draw(p, s, os.path.join(OUT, f"{s}_final.jpg"))
    summary = summarise(p, rms)
    summary["line_rms_px_per_marking"] = per_marking
    summary["params"] = p.tolist()
    json.dump(summary, open(os.path.join(OUT, "calibration.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=1))


def report(p, kps, title):
    dims, e, cams = split(p)
    world = pitch.keypoints(dims, e)
    print("==", title)
    print("  pitch", {k: round(float(v), 2) for k, v in pitch.dims_dict(dims).items()},
          "goal offsets", np.round(e, 2))
    for s in SIDES:
        names = list(kps[s])
        uv = project(cams[s], np.array([world[k] for k in names]))
        err = np.linalg.norm(uv - np.array([kps[s][k] for k in names]), axis=1)
        R, _ = cv2.Rodrigues(cams[s]["rvec"])
        print(f"  {s}: f={cams[s]['f']:.0f} centre={np.round(-R.T @ cams[s]['tvec'], 2)}"
              f" keypoint err px: median {np.nanmedian(err):.1f} max {np.nanmax(err):.1f}")
        print("    ", {n: round(float(v), 1) for n, v in zip(names, err)})


if __name__ == "__main__":
    main()
