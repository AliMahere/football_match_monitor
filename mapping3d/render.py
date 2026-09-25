"""
Step 5 - pictures and video of the 3D result.

    python mapping3d/render.py
        work/map3d/map3d.mp4        synced left | right views on top, top-down pitch below
        work/map3d/layout.png       where the cameras stand and what each one sees
        work/map3d/ball_height.png  ball height over time, stereo vs single camera
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pitch  # noqa: E402
from geometry import load_pair  # noqa: E402
from paths import FOOTAGE, TRACKS, WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "map3d")
CAL = json.load(open(os.path.join(WORK, "calib", "calibration.json")))
DIMS = np.array([CAL["pitch"][k] for k in pitch.DIM_NAMES])
PPM = 40            # pixels per metre on the top-down map
MARGIN = 2.0        # m around the pitch


def colour_for(i):
    rng = np.random.default_rng(i * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


class TopDown:
    def __init__(self):
        L, W = DIMS[0], DIMS[1]
        self.w = int((L + 2 * MARGIN) * PPM)
        self.h = int((W + 2 * MARGIN) * PPM)
        self.base = np.full((self.h, self.w, 3), (60, 120, 60), np.uint8)
        for poly in pitch.curves(DIMS).values():
            cv2.polylines(self.base, [self.px(poly[:, :2])], False, (240, 240, 240), 2,
                          cv2.LINE_AA)
        cv2.circle(self.base, tuple(self.px(np.array([[0, 0]]))[0]), 4, (240, 240, 240), -1)

    def px(self, xy):
        xy = np.atleast_2d(xy)
        return np.int32(np.c_[(xy[:, 0] + DIMS[0] / 2 + MARGIN) * PPM,
                              (DIMS[1] / 2 + MARGIN - xy[:, 1]) * PPM])   # far side on top

    def draw(self, players, ball, trail):
        im = self.base.copy()
        for p in players:
            c = (255, 200, 0) if len(p["cams"]) == 2 else (
                (80, 160, 255) if p["cams"] == ["left"] else (255, 120, 200))
            q = tuple(self.px([p["x"], p["y"]])[0])
            cv2.circle(im, q, 9, c, -1, cv2.LINE_AA)
            cv2.circle(im, q, 9, (20, 20, 20), 1, cv2.LINE_AA)
        for (x, y, z) in trail:
            cv2.circle(im, tuple(self.px([x, y])[0]), 2, (0, 220, 255), -1)
        if ball is not None and "x" in ball:
            q = tuple(self.px([ball["x"], ball["y"]])[0])
            r = int(6 + 6 * min(ball["z"], 3.0))       # bigger = higher
            col = (0, 0, 255) if ball["method"] == "stereo" else (0, 200, 255)
            cv2.circle(im, q, r, col, -1, cv2.LINE_AA)
            cv2.circle(im, q, r, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(im, f"z={ball['z']:.2f} m ({ball['method']})", (q[0] + 14, q[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        legend = [("seen by both (fused)", (255, 200, 0)), ("left only", (80, 160, 255)),
                  ("right only", (255, 120, 200)), ("ball: stereo", (0, 0, 255)),
                  ("ball: one camera", (0, 200, 255))]
        for i, (txt, c) in enumerate(legend):
            cv2.circle(im, (20, 20 + 22 * i), 7, c, -1)
            cv2.putText(im, txt, (34, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return im


def draw_boxes(img, frame_rec):
    for x1, y1, x2, y2, tid in frame_rec["players"]:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour_for(int(tid)), 4)
    if frame_rec["ball"]:
        x1, y1, x2, y2 = frame_rec["ball"]
        cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 30, (0, 0, 255), 5)
    return img


def video(players, ball, lag):
    TL = {f["frame"]: f for f in json.load(open(
        os.path.join(TRACKS, "final_left", "tracks.json")))["frames"]}
    TR = {f["frame"]: f for f in json.load(open(
        os.path.join(TRACKS, "final_right", "tracks.json")))["frames"]}
    td = TopDown()
    W = 1920
    th = int(td.h * W / td.w)
    H = 540 + th + (th % 2)
    capL = cv2.VideoCapture(os.path.join(FOOTAGE, "Left camera (stereo pair).mp4"))
    capR = cv2.VideoCapture(os.path.join(FOOTAGE, "Right camera (stereo pair).mp4"))
    f0 = players[0]["left_frame"]
    capL.set(cv2.CAP_PROP_POS_FRAMES, f0)
    capR.set(cv2.CAP_PROP_POS_FRAMES, f0 + lag)
    path = os.path.join(OUT, "map3d.mp4")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", "45", "-i", "-", "-c:v", "h264_nvenc",
                           "-cq", "23", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    trail = []
    for pf, b in zip(players, ball):
        okL, a = capL.read()
        okR, c = capR.read()
        if not (okL and okR):
            break
        a = cv2.resize(draw_boxes(a, TL[pf["left_frame"]]), (960, 540))
        c = cv2.resize(draw_boxes(c, TR[pf["right_frame"]]), (960, 540))
        for im, txt in ((a, f"LEFT frame {pf['left_frame']}"),
                        (c, f"RIGHT frame {pf['right_frame']}")):
            cv2.putText(im, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if "x" in b:
            trail.append((b["x"], b["y"], b["z"]))
        trail = trail[-60:]
        m = cv2.resize(td.draw(pf["players"], b, trail), (W, th))
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:540, :960], frame[:540, 960:] = a, c
        frame[540:540 + th] = m
        cv2.putText(frame, f"t = {pf['time_s']:.2f} s (left clock)", (W - 330, 540 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", path)


def layout():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    camL, camR = load_pair()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for poly in pitch.curves(DIMS).values():
        ax.plot(poly[:, 0], poly[:, 1], color="0.35", lw=1)
    xs, ys = np.meshgrid(np.linspace(-20, 20, 161), np.linspace(-14, 14, 113))
    G = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    sL, sR = camL.sees(G), camR.sees(G)
    ax.scatter(G[sL & ~sR, 0], G[sL & ~sR, 1], s=2, c="#4C8BF5", alpha=.35, label="left only")
    ax.scatter(G[sR & ~sL, 0], G[sR & ~sL, 1], s=2, c="#E8710A", alpha=.35, label="right only")
    ax.scatter(G[sL & sR, 0], G[sL & sR, 1], s=2, c="#1E8E3E", alpha=.6, label="both (stereo)")
    for cam, col in ((camL, "#4C8BF5"), (camR, "#E8710A")):
        ax.plot(*cam.C[:2], "^", color=col, ms=12, mec="k")
        ax.annotate(f"{cam.side}\n{cam.C[2]:.1f} m up", cam.C[:2], xytext=(8, -28),
                    textcoords="offset points")
    ax.annotate("", camR.C[:2], camL.C[:2], arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(0, camL.C[1] - 1.4, f"baseline {CAL['baseline_m']:.1f} m", ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", markerscale=5)
    ax.set_title(f"Recovered camera layout - pitch {DIMS[0]:.1f} x {DIMS[1]:.1f} m")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "layout.png"), dpi=120)


def height_plot(ball):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for m, c in (("stereo", "C3"), ("ballistic", "C0"), ("rolling", "C2"),
                 ("ground-short", "C7"), ("ground-edge", "C7")):
        pts = [(b["time_s"], b["z"]) for b in ball if b.get("method") == m]
        if pts:
            p = np.array(pts)
            ax.plot(p[:, 0], p[:, 1], ".", ms=4, color=c, label=m)
    ax.axhline(0.11, color="k", lw=.6, ls=":")
    ax.set_xlabel("time (s, left camera clock)")
    ax.set_ylabel("ball height z (m)")
    ax.set_title("Ball height. Dotted line: ball radius (ball on the grass)")
    ax.legend(ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ball_height.png"), dpi=120)


def main():
    players = json.load(open(os.path.join(OUT, "players.json")))
    ball = json.load(open(os.path.join(OUT, "ball.json")))
    lag = json.load(open(os.path.join(WORK, "sync", "sync.json")))["lag_frames"]
    layout()
    height_plot(ball)
    if "--no-video" not in sys.argv:
        video(players, ball, lag)


if __name__ == "__main__":
    main()
