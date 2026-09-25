# Requirement 3 — 3D mapping: options considered

Research notes, parked rather than finished. Nothing here is implemented yet.

## What makes this harder than the brief implies

The brief calls the two fixed cameras a stereo pair. They are not one in the usual
sense. They sit side by side at midfield and cover opposite halves of the pitch,
overlapping only around the centre circle, which was confirmed by finding the same
player at the right edge of one view and the left third of the other on the same frame
index.

Three consequences:

- Triangulation is only possible inside the narrow overlap band, so it cannot give
  positions for most of the pitch.
- If the two cameras share an optical centre and differ by rotation rather than
  translation, triangulation is impossible rather than merely noisy. Even a small real
  baseline is poorly conditioned at 30–50 m: disparities at that range become too small
  to separate.
- The ball is usually visible to only one camera, so stereo ball height is unavailable
  for most of the match whatever the baseline turns out to be.

The pitch is also small-sided and caged, not a full-size pitch, and the wide lenses bend
the touchlines visibly.

## The measurement that should come first

Whether a usable baseline exists is decidable from the footage. Take correspondences in
the centre-circle overlap at clearly different depths — a near player, a far player, a
point on a pitch line — and fit a single homography between the two views. If one
homography explains all of them, the cameras are effectively co-located and there is no
depth to recover. If near and far points need different mappings, a real baseline exists
and the essential matrix gives its direction.

This decides whether the requirement is answered with stereo or with monocular geometry.
Either answer is defensible once measured.

## Calibration

The obvious tools do not fit. Both TVCalib and PnLCalib work by leveraging the
standardised dimensions of a soccer pitch; TVCalib models the field as known geometric
primitives. Ours is roughly 40×25 m rather than 105×68, so a pretrained model would fit a
full-size field to a small one and return confident, wrong distances.

Three routes, in increasing risk:

1. **Manual homography with scale from a known object.** Hand-pick pitch correspondences
   and take metric scale from the goals, which are standardised at 3×2 m for five-a-side.
   Photogrammetry alone cannot resolve scale; a reference of known size supplies it.
2. **Plumb-line distortion correction first.** The pitch lines and the top edge of the
   barrier are straight in reality and visibly curved in the image. Solving for the
   radial distortion coefficients that make them straight is a standard self-calibration
   trick, and it matters because a homography assumes a pinhole camera and will fit worst
   at the frame edges — which is exactly where the overlap band sits.
3. **Adapt PnLCalib with our own field model.** Best quality if it works, most risk.

## Player positions

Players stand on the pitch, so z = 0 and a single homography per camera maps foot
position to metric pitch coordinates. This covers the whole pitch, needs no stereo, and
is what the SoccerNet Game State Reconstruction baseline does. Our detections already
give foot position as the bottom edge of the box.

## Ball height — four candidates

**Ballistic fitting.** A ball in flight follows a parabola, so a 3D trajectory whose
reprojection matches the observed 2D track can be solved with gravity fixing the scale.
This is the classic physics-based approach. The tracking work already detects when the
ball is airborne: the physics motion model fits a parabola to recent centres and exposes
the fitted vertical acceleration, which turns clearly positive in flight.

**Ball size prior.** Apparent diameter against known real diameter gives depth directly.
This is the SoccerNet-v3D baseline. Cheap and available across the whole pitch, but noisy
when the ball is only about 34 px across.

**Stereo triangulation.** Overlap band only, and only if the baseline measurement above
comes out favourable.

**Shadow geometry.** The footage is sunlit. A ball's shadow on the pitch together with
its image position fixes height geometrically. Cheap where shadows are visible, not
dependable enough to build on alone.

## Prior work worth citing but not usable

- **Where Is The Ball** (CVPR-W 2025) estimates a 3D trajectory from 2D tracking alone,
  independent of camera setup, which is exactly our problem. Code is announced but not
  released.
- **Real-time localization of a soccer ball from a single camera** reports centimetre
  accuracy but is evaluated on a proprietary dataset with no code.

## Validation

**ISSIA-3D** provides ground-truth 3D ball positions from fixed cameras calibrated from
field lines, which is the same problem we have. Running our pipeline against it would
turn a claim into a measured error.

**Checks that need no labels.** Projected running speeds expose identity errors and
scale errors at once, since a track implying 40 km/h is wrong either way. Recovered pitch
dimensions should also be self-consistent between the two cameras.

## Recommended order when this is picked up

1. Measure the baseline and settle whether stereo is available at all.
2. Correct radial distortion using the straight pitch lines.
3. Per-camera homography with scale from the goals.
4. Players via the z = 0 foot assumption.
5. Ball height by ballistic fitting, using the existing airborne signal, with
   triangulation in the overlap as a cross-check if the baseline permits it.
