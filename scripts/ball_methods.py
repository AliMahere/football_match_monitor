"""Ball tracking strategies, so they can be compared rather than assumed.

All of them consume the same per-frame candidate detections and differ only in
how they predict where the ball goes next and which candidates they believe.

  sweep     no memory at all. Every frame is searched from scratch and the
            highest-scoring candidate wins. The honest baseline.
  velocity  constant velocity. Position plus last step.
  kalman    constant-acceleration Kalman filter, which smooths noisy centres and
            coasts through short misses.
  physics   ballistic model. A ball that is not being touched is in free fall, so
            vertical image acceleration is roughly constant and positive. Fits a
            parabola over recent history to predict, and rejects candidates that
            would demand an impossible acceleration.

The physics model matters here for a specific reason: the recurring false
positive on this footage is the penalty spot, a white mark the size of a ball
that never moves. A model that expects ballistic motion rejects a stationary
candidate by construction instead of having to be taught it.
"""

import numpy as np


class BallTracker:
    """Common interface: predict() gives a search centre, update() accepts a centre."""

    name = "base"

    def __init__(self, max_speed=260.0):
        # Pixels per frame. A struck ball crosses this 4K frame in well under a
        # second, but a jump larger than this between frames is not the same ball.
        self.max_speed = max_speed
        self.position = None
        self.history = []
        self.misses = 0

    def predict(self):
        return self.position

    def plausible(self, candidate):
        predicted = self.predict()
        if predicted is None:
            return True
        allowed = self.max_speed * (1 + self.misses)
        return float(np.linalg.norm(np.asarray(candidate) - predicted)) <= allowed

    def update(self, centre):
        if centre is None:
            self.misses += 1
            return
        centre = np.asarray(centre, dtype=float)
        self.position = centre
        self.history.append(centre)
        self.history = self.history[-12:]
        self.misses = 0

    def reset(self):
        self.position = None
        self.history = []
        self.misses = 0


class SweepTracker(BallTracker):
    name = "sweep"

    def predict(self):
        return None  # forces a full-frame search every frame

    def plausible(self, candidate):
        return True


class VelocityTracker(BallTracker):
    name = "velocity"

    def predict(self):
        if len(self.history) < 2:
            return self.position
        return self.history[-1] + (self.history[-1] - self.history[-2])


class KalmanTracker(BallTracker):
    """Constant acceleration in both axes."""

    name = "kalman"

    def __init__(self, max_speed=260.0, process=6.0, measurement=9.0):
        super().__init__(max_speed)
        self.state = None            # [x, y, vx, vy, ax, ay]
        self.covariance = None
        self.process = process
        self.measurement = measurement

    def _transition(self):
        transition = np.eye(6)
        transition[0, 2] = transition[1, 3] = 1.0
        transition[0, 4] = transition[1, 5] = 0.5
        transition[2, 4] = transition[3, 5] = 1.0
        return transition

    def predict(self):
        if self.state is None:
            return None
        return (self._transition() @ self.state)[:2]

    def update(self, centre):
        if centre is None:
            self.misses += 1
            if self.state is not None:
                transition = self._transition()
                self.state = transition @ self.state
                self.covariance = (
                    transition @ self.covariance @ transition.T
                    + np.eye(6) * self.process
                )
            return

        centre = np.asarray(centre, dtype=float)
        if self.state is None:
            self.state = np.array([centre[0], centre[1], 0, 0, 0, 0], dtype=float)
            self.covariance = np.eye(6) * 100.0
        else:
            transition = self._transition()
            predicted = transition @ self.state
            covariance = (
                transition @ self.covariance @ transition.T + np.eye(6) * self.process
            )
            observation = np.zeros((2, 6))
            observation[0, 0] = observation[1, 1] = 1.0
            innovation = centre - observation @ predicted
            innovation_cov = (
                observation @ covariance @ observation.T + np.eye(2) * self.measurement
            )
            gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
            self.state = predicted + gain @ innovation
            self.covariance = (np.eye(6) - gain @ observation) @ covariance

        self.position = self.state[:2].copy()
        self.history.append(centre)
        self.history = self.history[-12:]
        self.misses = 0


class PhysicsTracker(BallTracker):
    """Fits a parabola to recent centres: constant velocity across, constant
    acceleration down. Rejects candidates demanding impossible acceleration."""

    name = "physics"

    def __init__(self, max_speed=260.0, fit_window=6, max_accel=40.0):
        super().__init__(max_speed)
        self.fit_window = fit_window
        self.max_accel = max_accel
        self.last_fit = None

    def _fit(self):
        points = self.history[-self.fit_window :]
        if len(points) < 3:
            return None
        times = np.arange(len(points), dtype=float)
        array = np.asarray(points)
        # x is linear in time, y is quadratic: the signature of free flight.
        x_coeffs = np.polyfit(times, array[:, 0], 1)
        y_coeffs = np.polyfit(times, array[:, 1], 2)
        return x_coeffs, y_coeffs, len(points)

    def predict(self):
        fit = self._fit()
        if fit is None:
            if len(self.history) >= 2:
                return self.history[-1] + (self.history[-1] - self.history[-2])
            return self.position
        x_coeffs, y_coeffs, count = fit
        step = count + self.misses
        self.last_fit = fit
        return np.array([np.polyval(x_coeffs, step), np.polyval(y_coeffs, step)])

    def plausible(self, candidate):
        if not super().plausible(candidate):
            return False
        fit = self._fit()
        if fit is None:
            return True
        # 2*a is the per-frame vertical acceleration implied by the parabola.
        _, y_coeffs, _ = fit
        return abs(2.0 * y_coeffs[0]) <= self.max_accel

    def airborne(self):
        """A clear positive vertical acceleration means the ball is in flight,
        which is the cue the 3D stage needs for height above the pitch."""
        fit = self._fit()
        if fit is None:
            return False
        _, y_coeffs, _ = fit
        return 0.5 < 2.0 * y_coeffs[0] <= self.max_accel


TRACKERS = {
    "sweep": SweepTracker,
    "velocity": VelocityTracker,
    "kalman": KalmanTracker,
    "physics": PhysicsTracker,
}
