"""
Lens distortion: the division model (Fitzgibbon 2001), good for wide-angle lenses.

    undistort:  x_u = c + (x_d - c) / (1 + l1 * r^2 + l2 * r^4),   r = |x_d - c| / S

with c the distortion centre and S = 1920 px (half the 4K width) to keep l1, l2 near 1.
Undistorting is closed-form; distorting (needed to draw 3D points back into the real
image) is a 1-D radial root-find, done with a few Newton steps.

After undistortion the camera is an ideal pinhole, so every straight line in the world
is straight in the image. That is exactly the property plumb-line calibration enforces.
"""
import numpy as np

S = 1920.0


class Lens:
    def __init__(self, l1=0.0, l2=0.0, cx=1920.0, cy=1080.0):
        self.l1, self.l2, self.c = float(l1), float(l2), np.array([cx, cy], float)

    @classmethod
    def from_list(cls, v):
        return cls(*v)

    def to_list(self):
        return [self.l1, self.l2, float(self.c[0]), float(self.c[1])]

    def undistort(self, xd):
        xd = np.asarray(xd, float)
        d = (xd - self.c) / S
        r2 = (d ** 2).sum(-1, keepdims=True)
        return self.c + S * d / (1 + self.l1 * r2 + self.l2 * r2 ** 2)

    def r_max(self):
        """Largest distorted radius where the model is still monotonic (r_u grows with r_d)
        and the denominator positive. Beyond it the model folds back."""
        r = np.linspace(0, 3, 30001)
        g = 1 + self.l1 * r ** 2 + self.l2 * r ** 4
        ru = np.where(g > 1e-6, r / np.maximum(g, 1e-6), 1e12)
        bad = np.flatnonzero((g <= 1e-6) | (np.diff(ru, prepend=-1) <= 0))
        return r[bad[0] - 1] if len(bad) else 3.0

    def distort(self, xu):
        """Ideal-pinhole pixels -> real (distorted) pixels.

        Solves r_u = r_d / (1 + l1 r_d^2 + l2 r_d^4) for r_d on [0, r_max], where the right
        side is increasing, so the root is unique. Bisection first (it can't diverge), then
        a few Newton steps to polish. A plain Newton started at r_d = r_u - an earlier
        version - diverged in the outer ~5 % of the image, where r_u is past r_max."""
        xu = np.asarray(xu, float)
        u = (xu - self.c) / S
        ru = np.sqrt((u ** 2).sum(-1))
        f = lambda r: r / (1 + self.l1 * r ** 2 + self.l2 * r ** 4)
        lo, hi = np.zeros_like(ru), np.full_like(ru, self.r_max())
        for _ in range(45):                      # bisection: 45 halvings -> ~1e-13
            mid = (lo + hi) / 2
            below = f(mid) < ru
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        rd = (lo + hi) / 2
        for _ in range(2):                       # Newton polish
            g = 1 + self.l1 * rd ** 2 + self.l2 * rd ** 4
            dg = 2 * self.l1 * rd + 4 * self.l2 * rd ** 3
            df = (g - rd * dg) / g ** 2
            rd = rd - (rd / g - ru) / np.where(np.abs(df) < 1e-9, 1e-9, df)
        rd = np.clip(rd, 0, self.r_max())
        scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
        out = self.c + S * u * scale[..., None]
        # rays whose undistorted radius is beyond what the lens can image: not visible
        out[ru > f(self.r_max()) * (1 + 1e-9)] = np.nan
        return out
