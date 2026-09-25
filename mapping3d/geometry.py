"""
Camera geometry built on the calibration (work/calib/calibration.json).

    cam = Camera.load("left")
    cam.to_pixel(X)            world (N,3) -> real (distorted) image pixels (N,2)
    cam.ray(px)                pixels -> (centre, unit directions) in world coords
    cam.to_plane(px, z=0)      pixels -> world point where the ray meets height z
    triangulate(camL, pxL, camR, pxR) -> 3D points + gap between the two rays

World frame: metres, origin at the centre spot, x along the pitch towards the right
camera's goal, y towards the far touchline, z up.
"""
import json
import os

import cv2
import numpy as np

from lens import Lens
from paths import WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(WORK, "calib", "calibration.json")
W_IMG, H_IMG = 3840, 2160


class Camera:
    def __init__(self, side, f, rvec, tvec, lens):
        self.side, self.f, self.lens = side, float(f), lens
        self.rvec = np.asarray(rvec, float)
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.t = np.asarray(tvec, float)
        self.C = -self.R.T @ self.t
        self.c = np.array([W_IMG / 2, H_IMG / 2])   # pinhole principal point

    @classmethod
    def load(cls, side, path=CALIB):
        c = json.load(open(path))["cameras"][side]
        return cls(side, c["K_undistorted"][0][0], c["rvec"], c["tvec"],
                   Lens.from_list(c["lens_division"]))

    # --- projection -----------------------------------------------------------------
    def to_undistorted(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xc = X @ self.R.T + self.t
        uv = self.f * Xc[:, :2] / Xc[:, 2:3] + self.c
        uv[Xc[:, 2] <= 0] = np.nan
        return uv

    def to_pixel(self, X):
        return self.lens.distort(self.to_undistorted(X))

    def sees(self, X, margin=20):
        """True where the world point lands inside the real image."""
        px = self.to_pixel(X)
        return ((px[:, 0] > margin) & (px[:, 0] < W_IMG - margin) &
                (px[:, 1] > margin) & (px[:, 1] < H_IMG - margin))

    # --- back-projection ------------------------------------------------------------
    def ray(self, px):
        u = self.lens.undistort(np.atleast_2d(np.asarray(px, float)))
        d = np.c_[(u - self.c) / self.f, np.ones(len(u))] @ self.R   # = R^T [x y 1]
        return self.C, d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_plane(self, px, z=0.0):
        C, d = self.ray(px)
        s = (z - C[2]) / d[:, 2]
        P = C + s[:, None] * d
        P[s <= 0] = np.nan
        return P

    def metres_per_pixel(self, X):
        """Approximate size of one image pixel at world point X (for error budgets)."""
        dist = np.linalg.norm(np.atleast_2d(X) - self.C, axis=1)
        u = self.to_undistorted(X)
        r2 = (((self.lens.distort(u) - self.lens.c) / 1920.0) ** 2).sum(1)
        mag = 1 / (1 + self.lens.l1 * r2 + self.lens.l2 * r2 ** 2)   # undist px per px
        return dist * mag / self.f


def triangulate(camA, pxA, camB, pxB):
    """Closest-approach midpoint of two rays per pair. Returns (points (N,3), gap m)."""
    CA, dA = camA.ray(pxA)
    CB, dB = camB.ray(pxB)
    w0 = CA - CB
    a, b, c = (dA * dA).sum(1), (dA * dB).sum(1), (dB * dB).sum(1)
    d, e = (dA * w0).sum(1), (dB * w0).sum(1)
    den = a * c - b * b
    sA = (b * e - c * d) / den
    sB = (a * e - b * d) / den
    PA, PB = CA + sA[:, None] * dA, CB + sB[:, None] * dB
    return (PA + PB) / 2, np.linalg.norm(PA - PB, axis=1)


def load_pair():
    return Camera.load("left"), Camera.load("right")
