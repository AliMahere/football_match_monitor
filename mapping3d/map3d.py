"""
Step 4 - real-world positions of players and ball, in metres, from the synced pair.

    python mapping3d/map3d.py   -> work/map3d/{players,ball}.json, summary.json

Input: the tracker output from requirements 1-2 (tracks.json per camera: player boxes with
IDs, ball box per frame), the calibration, and the measured lag.

PLAYERS (whole pitch, both cameras)
  A standing player's feet are on the grass (z = 0), so one camera is enough: the ray
  through the bottom-centre of the box meets the ground plane at the player's position.
  In the overlap both cameras see the same player; the two estimates are matched
  (Hungarian, 1.5 m gate) and averaged. There, stereo also triangulates the TOP of the
  box, giving each player's height: an independent check of the calibration, since
  nothing in the calibration knew how tall anyone is.

BALL (height included)
  * stereo: when both cameras see the ball at the same moment (overlap), its 3D position
    is the closest point between the two rays. No assumption about height needed.
  * one camera: a single ray does not fix depth, so two motion models compete on each
    short window of the track (0.4 s):
        rolling   - ball centre at z = r (0.11 m), CONSTANT velocity on the ground
                    (4 params). Rolling friction is ~0.5 m/s^2, i.e. nothing over 0.4 s.
        ballistic - free flight under gravity, x(t) = x0 + v t + g t^2 / 2 (6 params)
    Each is scored by reprojection error in pixels. Gravity is what makes the ballistic
    model identifiable from one view. Ballistic wins only if it explains the pixels
    clearly better AND the fitted flight is physical (above the grass, below 12 m).
    Why constant velocity: an earlier version let the rolling ball accelerate freely
    (6 params, like ballistic). A low hop seen from one camera then fitted just as well
    as a roll with an unphysical acceleration along the viewing ray, and no flight was
    ever detected. Checked against stereo in the overlap (see summary.json).
  * the ball's apparent size (22 cm) gives a third, noisy distance estimate; it is
    reported as a sanity check only (the ball is ~20-40 px wide, so +-2 px is +-10 %).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from geometry import H_IMG, W_IMG, load_pair, triangulate  # noqa: E402
from paths import TRACKS, WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "map3d")
FPS = 45.0
G = np.array([0, 0, -9.81])
BALL_R = 0.11          # m (size-5 ball, 22 cm diameter)
WIN = 18               # frames per model-selection window (0.4 s)


def load_tracks(side):
    d = json.load(open(os.path.join(TRACKS, f"final_{side}", "tracks.json")))
    return {f["frame"]: f for f in d["frames"]}


# ------------------------------------------------------------------------- players
def player_ground(cam, players):
    """[(track id, x, y, foot_px, top_px)] for boxes whose feet are inside the image."""
    out = []
    for x1, y1, x2, y2, tid in players:
        if y2 > H_IMG - 10:        # feet cut off by the image border
            continue
        foot = np.array([[(x1 + x2) / 2, y2]])
        P = cam.to_plane(foot, 0.0)[0]
        if np.isnan(P).any():
            continue
        out.append((int(tid), P[0], P[1], foot[0], np.array([(x1 + x2) / 2, y1])))
    return out


def fuse_players(camL, camR, pl, pr):
    """Merge the two cameras' player lists. Returns list of dicts."""
    res, used_r = [], set()
    inL = [p for p in pl if camR.sees(np.array([[p[1], p[2], 0]]), 60)[0]]
    inR = [p for p in pr if camL.sees(np.array([[p[1], p[2], 0]]), 60)[0]]
    matches = {}
    if inL and inR:
        A = np.array([[p[1], p[2]] for p in inL])
        B = np.array([[p[1], p[2]] for p in inR])
        D = np.linalg.norm(A[:, None] - B[None], axis=-1)
        ri, ci = linear_sum_assignment(D)
        for i, j in zip(ri, ci):
            if D[i, j] < 1.5:
                matches[inL[i][0]] = (inR[j], float(D[i, j]))
    for p in pl:
        rec = {"left_id": p[0], "x": p[1], "y": p[2], "cams": ["left"]}
        if p[0] in matches:
            q, gap = matches[p[0]]
            used_r.add(q[0])
            top, _ = triangulate(camL, p[4][None], camR, q[4][None])
            rec.update({"right_id": q[0], "x": (p[1] + q[1]) / 2, "y": (p[2] + q[2]) / 2,
                        "cams": ["left", "right"], "cam_disagreement_m": gap,
                        "stereo_height_m": float(top[0, 2])})
        res.append(rec)
    for q in pr:
        if q[0] not in used_r:
            res.append({"right_id": q[0], "x": q[1], "y": q[2], "cams": ["right"]})
    for r in res:
        r["x"], r["y"] = round(float(r["x"]), 3), round(float(r["y"]), 3)
    return res


# ------------------------------------------------------------------------- ball, 1 cam
def ball_px(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2]), max(x2 - x1, y2 - y1)


def fit_rolling(cam, t, px):
    """Ground-plane motion at z = r, constant velocity (4 params)."""
    P = cam.to_plane(px, BALL_R)[:, :2]
    if np.isnan(P).any():
        return None
    A = np.c_[np.ones_like(t), t]
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    X = np.c_[A @ coef, np.full(len(t), BALL_R)]
    err = np.linalg.norm(cam.to_pixel(X) - px, axis=1)
    return {"X": X, "rms": float(np.sqrt(np.mean(err ** 2)))}


def fit_ballistic(cam, t, px, seed):
    """Free flight under gravity (6 params: start position + velocity)."""
    def res(q):
        X = q[:3] + np.outer(t, q[3:]) + 0.5 * np.outer(t ** 2, G)
        uv = cam.to_pixel(X)
        return np.nan_to_num(uv - px, nan=1e3).ravel()
    best = None
    for z0 in (0.3, 1.5, 4.0):   # a few starting heights along the first ray
        C, d = cam.ray(px[:1])
        s = max((z0 - C[2]) / d[0, 2], 1.0) if d[0, 2] < 0 else 10.0
        x0 = C + s * d[0]
        v0 = (seed[-1] - seed[0]) / max(t[-1] - t[0], 1e-3)
        x0[2] = np.clip(x0[2], 0.05, 11.9)
        lo = [-60, -40, 0.0, -40, -40, -25]
        hi = [60, 40, 12.0, 40, 40, 25]
        r = least_squares(res, np.clip(np.r_[x0, v0], lo, hi), loss="soft_l1", f_scale=3.0,
                          bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    q = best.x
    X = q[:3] + np.outer(t, q[3:]) + 0.5 * np.outer(t ** 2, G)
    err = np.linalg.norm(best.fun.reshape(-1, 2), axis=1)
    return {"X": X, "rms": float(np.sqrt(np.mean(err ** 2)))}


def ball_monocular(cam, dets):
    """dets: {frame: box}. Returns {frame: (X, model, rms)} from sliding windows."""
    frames = sorted(dets)
    votes = {f: [] for f in frames}
    # windows of consecutive detections (gaps of <=2 frames allowed)
    runs, cur = [], [frames[0]] if frames else []
    for f in frames[1:]:
        if f - cur[-1] <= 3:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    if cur:
        runs.append(cur)
    for run in runs:
        if len(run) < 8:
            for f in run:   # too short to fit motion: assume on the ground
                px, _ = ball_px(dets[f])
                X = cam.to_plane(px[None], BALL_R)[0]
                votes[f].append((X, "ground-short", np.nan, 0.0))
            continue
        step = WIN // 2
        for s in range(0, max(1, len(run) - WIN + 1), step):
            w = run[s:s + WIN]
            t = (np.array(w) - w[0]) / FPS
            px = np.array([ball_px(dets[f])[0] for f in w])
            roll = fit_rolling(cam, t, px)
            if roll is None:
                continue
            fly = fit_ballistic(cam, t, px, roll["X"])
            # a flight must fit WELL, not merely better: a window that straddles a kick
            # breaks both models, and "less bad" produced a false 4 m flight at 46 s
            airborne = (fly["rms"] < roll["rms"] / 1.6 and roll["rms"] > 2.0
                        and fly["rms"] < 6.0
                        and fly["X"][:, 2].max() > 0.3 and fly["X"][:, 2].min() > -0.15)
            m = fly if airborne else roll
            for f, X in zip(w, m["X"]):
                votes[f].append((X, "ballistic" if airborne else "rolling", m["rms"],
                                 roll["rms"] - fly["rms"]))
    out = {}
    for f, v in votes.items():
        if not v:
            px, _ = ball_px(dets[f])
            out[f] = (cam.to_plane(px[None], BALL_R)[0], "ground-edge", np.nan)
            continue
        fly = [x for x in v if x[1] == "ballistic"]
        pick = fly if len(fly) * 2 >= len(v) else [x for x in v if x[1] != "ballistic"] or v
        X = np.median(np.array([x[0] for x in pick]), axis=0)
        out[f] = (X, pick[0][1], float(np.nanmedian([x[2] for x in pick])))
    return out


def size_distance(cam, box):
    """Distance to the ball from its apparent size (pinhole, lens-corrected)."""
    px, w = ball_px(box)
    r2 = (((px - cam.lens.c) / 1920.0) ** 2).sum()
    mag = 1 / (1 + cam.lens.l1 * r2 + cam.lens.l2 * r2 ** 2)
    return cam.f * 2 * BALL_R / (w * mag)


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    camL, camR = load_pair()
    lag = json.load(open(os.path.join(WORK, "sync", "sync.json")))["lag_frames"]
    TL, TR = load_tracks("left"), load_tracks("right")
    frames = [t for t in sorted(TL) if t + lag in TR]
    print(f"lag {lag}: {len(frames)} synced frames "
          f"(left {frames[0]}-{frames[-1]}, right {frames[0] + lag}-{frames[-1] + lag})")

    # players
    players, heights, gaps = [], [], []
    for t in frames:
        pl = player_ground(camL, TL[t]["players"])
        pr = player_ground(camR, TR[t + lag]["players"])
        fused = fuse_players(camL, camR, pl, pr)
        for r in fused:
            if "stereo_height_m" in r:
                heights.append(r["stereo_height_m"])
                gaps.append(r["cam_disagreement_m"])
        players.append({"left_frame": t, "right_frame": t + lag,
                        "time_s": round(t / FPS, 4), "players": fused})
    json.dump(players, open(os.path.join(OUT, "players.json"), "w"))

    # ball: per camera monocular, then stereo where both see it
    detL = {t: TL[t]["ball"] for t in frames if TL[t]["ball"]}
    detR = {t: TR[t + lag]["ball"] for t in frames if TR[t + lag]["ball"]}
    monoL = ball_monocular(camL, detL) if detL else {}
    monoR = ball_monocular(camR, detR) if detR else {}
    ball, stereo_rows = [], []
    for t in frames:
        rec = {"left_frame": t, "right_frame": t + lag, "time_s": round(t / FPS, 4)}
        if t in detL and t in detR:
            pL, _ = ball_px(detL[t])
            pR, _ = ball_px(detR[t])
            X, gap = triangulate(camL, pL[None], camR, pR[None])
            X, gap = X[0], float(gap[0])
            if gap < 0.6:
                rec.update({"x": X[0], "y": X[1], "z": X[2], "method": "stereo",
                            "ray_gap_m": gap})
                mono = [m for m in (monoL.get(t), monoR.get(t)) if m]
                stereo_rows.append({"t": t, "stereo": X.tolist(), "gap": gap,
                                    "mono": [m[0].tolist() for m in mono],
                                    "mono_models": [m[1] for m in mono],
                                    "size_dist": [size_distance(camL, detL[t]),
                                                  size_distance(camR, detR[t])],
                                    "ray_dist": [float(np.linalg.norm(X - camL.C)),
                                                 float(np.linalg.norm(X - camR.C))]})
        if "method" not in rec:
            cands = [(m, side) for m, side in ((monoL.get(t), "left"), (monoR.get(t), "right"))
                     if m]
            if cands:
                (X, model, rms), side = cands[0]
                rec.update({"x": X[0], "y": X[1], "z": X[2], "method": model,
                            "camera": side, "fit_rms_px": rms})
        for k in ("x", "y", "z"):
            if k in rec:
                rec[k] = round(float(rec[k]), 3)
        ball.append(rec)
    json.dump(ball, open(os.path.join(OUT, "ball.json"), "w"))
    json.dump(stereo_rows, open(os.path.join(OUT, "ball_stereo_check.json"), "w"))

    # summary + validation numbers
    methods = {}
    for b in ball:
        methods[b.get("method", "not seen")] = methods.get(b.get("method", "not seen"), 0) + 1
    sr = stereo_rows
    val = {}
    if sr:
        S = np.array([r["stereo"] for r in sr])
        val["stereo_ball_frames"] = len(sr)
        val["stereo_ball_height_m"] = {"median": float(np.median(S[:, 2])),
                                       "p90": float(np.percentile(S[:, 2], 90)),
                                       "max": float(S[:, 2].max())}
        val["stereo_ray_gap_m_median"] = float(np.median([r["gap"] for r in sr]))
        mono_err = [np.linalg.norm(np.array(m) - np.array(r["stereo"]))
                    for r in sr for m in r["mono"]]
        val["mono_vs_stereo_error_m"] = {"median": float(np.median(mono_err)),
                                         "p90": float(np.percentile(mono_err, 90))}
        sz = np.array([[r["size_dist"][i], r["ray_dist"][i]] for r in sr for i in (0, 1)])
        val["size_prior_distance_error_pct_median"] = float(
            np.median(np.abs(sz[:, 0] / sz[:, 1] - 1) * 100))
        # single-camera "is it in the air?" judged against stereo (truth: z > 0.4 m)
        truth = np.array([r["stereo"][2] > 0.4 for r in sr for _ in r["mono_models"]])
        pred = np.array([m == "ballistic" for r in sr for m in r["mono_models"]])
        val["mono_airborne_vs_stereo"] = {
            "frames": int(len(truth)), "stereo_airborne": int(truth.sum()),
            "mono_said_airborne": int(pred.sum()), "both": int((truth & pred).sum()),
            "recall": float((truth & pred).sum() / max(truth.sum(), 1)),
            "precision": float((truth & pred).sum() / max(pred.sum(), 1))}
        err_air = [np.linalg.norm(np.array(m) - np.array(r["stereo"]))
                   for r in sr for m in r["mono"] if r["stereo"][2] > 0.4]
        err_gnd = [np.linalg.norm(np.array(m) - np.array(r["stereo"]))
                   for r in sr for m in r["mono"] if r["stereo"][2] <= 0.4]
        val["mono_vs_stereo_error_m_ball_on_ground"] = float(np.median(err_gnd))
        val["mono_vs_stereo_error_m_ball_in_air"] = float(np.median(err_air)) if err_air else None
    speeds = player_speeds(players)
    summary = {"lag_frames": lag, "synced_frames": len(frames),
               "seconds": len(frames) / FPS,
               "ball_frames_by_method": methods,
               "players_per_frame": float(np.mean([len(p["players"]) for p in players])),
               "players_seen_by_both_per_frame": float(np.mean(
                   [sum(len(r["cams"]) == 2 for r in p["players"]) for p in players])),
               "player_stereo_height_m": ({"median": float(np.median(heights)),
                                           "p10": float(np.percentile(heights, 10)),
                                           "p90": float(np.percentile(heights, 90)),
                                           "n": len(heights)} if heights else None),
               "player_left_vs_right_ground_disagreement_m": (
                   {"median": float(np.median(gaps)), "p90": float(np.percentile(gaps, 90))}
                   if gaps else None),
               "player_speed_kmh": speeds, "ball_validation": val}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


def player_speeds(players):
    """Running speed per camera track, smoothed over 0.5 s - a label-free sanity check:
    a scale error or an identity switch shows up as impossible speeds."""
    tracks = {}
    for fr in players:
        for r in fr["players"]:
            for side in ("left", "right"):
                if f"{side}_id" in r:
                    tracks.setdefault((side, r[f"{side}_id"]), []).append(
                        (fr["time_s"], r["x"], r["y"]))
    sp = []
    for pts in tracks.values():
        a = np.array(pts)
        if len(a) < 30:
            continue
        k = 22   # 0.5 s
        d = np.linalg.norm(a[k:, 1:] - a[:-k, 1:], axis=1)
        dt = a[k:, 0] - a[:-k, 0]
        sp.extend((d / dt * 3.6)[dt < 0.6].tolist())
    sp = np.array(sp)
    return {"median": float(np.median(sp)), "p95": float(np.percentile(sp, 95)),
            "p99": float(np.percentile(sp, 99)), "max": float(sp.max())} if len(sp) else None


if __name__ == "__main__":
    main()
