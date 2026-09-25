"""
Parametric model of the pitch, in metres.

World frame (shared by both cameras):
    origin  = centre spot
    x axis  = along the pitch, towards the goal seen by the RIGHT camera
    y axis  = across the pitch, towards the FAR touchline (the one along the barrier
              at the top of both images)
    z axis  = up. The grass is z = 0.

The pitch size is NOT known in advance (it is a small-sided caged pitch), so every
dimension is a free parameter, estimated during calibration. Only the goal frame is
assumed: 3 m wide x 2 m high (futsal / 5-a-side standard). That is what fixes metric
scale; everything else is measured.
"""
import numpy as np

# name -> initial guess (m). Refined by calibrate.py.
DIM_NAMES = ["L", "W", "box_d", "box_w", "spot", "R", "D_r", "D_c"]
DIM_INIT = np.array([34.0, 22.5, 5.8, 8.0, 5.0, 2.5, 2.0, 0.0])
GOAL_W, GOAL_H = 3.0, 2.0


def dims_dict(d):
    return dict(zip(DIM_NAMES, d))


def curves(d, n=240):
    """Every painted marking as a polyline (N x 3, z = 0). Returns {name: array}."""
    L, W, bd, bw, sp, R, Dr, Dc = d
    hx, hy = L / 2, W / 2
    seg = lambda a, b: np.linspace(a, b, n)
    out = {
        "far_touch": seg([-hx, hy], [hx, hy]),
        "goal_line_L": seg([-hx, -hy], [-hx, hy]),
        "goal_line_R": seg([hx, -hy], [hx, hy]),
        "halfway": seg([0, -hy], [0, hy]),
    }
    t = np.linspace(0, 2 * np.pi, 2 * n)
    out["circle"] = np.c_[R * np.cos(t), R * np.sin(t)]
    for side, s in (("L", -1), ("R", 1)):
        gx, fx = s * hx, s * (hx - bd)            # goal line x, box front x
        out[f"box_front_{side}"] = seg([fx, -bw / 2], [fx, bw / 2])
        out[f"box_top_{side}"] = seg([gx, bw / 2], [fx, bw / 2])
        out[f"box_bot_{side}"] = seg([gx, -bw / 2], [fx, -bw / 2])
        # "D": arc on the pitch side of the box front, centred on the pitch axis
        # (D_c > 0 moves its centre from the box front towards the goal)
        cx = fx + s * Dc
        a = np.linspace(-np.pi, np.pi, 2 * n)
        arc = np.c_[cx + Dr * np.cos(a), Dr * np.sin(a)]
        out[f"D_{side}"] = arc[(arc[:, 0] - fx) * -s >= 0]
    return {k: np.c_[v, np.zeros(len(v))] for k, v in out.items()}


def keypoints(d, e_goal=(0.0, 0.0)):
    """Named 3D points. e_goal = how far each goal stands behind its goal line."""
    L, W, bd, bw, sp, R, Dr, Dc = d
    hx, hy = L / 2, W / 2
    p = {
        "centre": (0, 0, 0), "half_far": (0, hy, 0),
        "half_circ_far": (0, R, 0), "half_circ_near": (0, -R, 0),
        "corner_far_L": (-hx, hy, 0), "corner_far_R": (hx, hy, 0),
    }
    for side, s, e in (("L", -1, e_goal[0]), ("R", 1, e_goal[1])):
        gx, fx = s * hx, s * (hx - bd)
        p[f"box_in_far_{side}"] = (fx, bw / 2, 0)
        p[f"box_in_near_{side}"] = (fx, -bw / 2, 0)
        p[f"box_gl_far_{side}"] = (gx, bw / 2, 0)
        p[f"box_gl_near_{side}"] = (gx, -bw / 2, 0)
        p[f"spot_{side}"] = (s * (hx - sp), 0, 0)
        px = s * (hx + e)
        p[f"post_far_base_{side}"] = (px, GOAL_W / 2, 0)
        p[f"post_far_top_{side}"] = (px, GOAL_W / 2, GOAL_H)
        p[f"post_near_base_{side}"] = (px, -GOAL_W / 2, 0)
        p[f"post_near_top_{side}"] = (px, -GOAL_W / 2, GOAL_H)
    return {k: np.array(v, float) for k, v in p.items()}
