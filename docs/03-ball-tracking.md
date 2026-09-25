# 03 — Tracking the ball

Code: `tracking/track.py` (`BallSearcher`), `tracking/ball_methods.py`,
`tracking/eval_ball.py`. Requirement 2.

## Goal

Follow a ball that is about 34 px across in a 3840 × 2160 frame, moves fast, and is
regularly hidden by players.

## What we tried

**Detection at native scale.** The fine-tuned model from `01-detection.md` only sees the
ball at its real size inside a native-resolution crop. Searching every tile of every
frame works but is slow, so the searcher first looks in a 768 px window around where the
ball is expected, and only sweeps the whole frame in tiles when that window comes up
empty. The window grows with each consecutive miss, and after five misses in a row the
track is dropped and rebuilt from the next sweep.

**Four ways of predicting where the ball goes**, behind one interface in
`ball_methods.py`:

- **sweep** — no memory, a full search every frame. The baseline.
- **velocity** — last position plus last step.
- **kalman** — constant-acceleration Kalman filter over position, velocity and
  acceleration.
- **physics** — a ball in free flight moves linearly across the image and
  quadratically up and down. Fits a parabola to the last six positions to predict, and
  rejects any candidate that would need an impossible acceleration. This rejects a
  stationary mark such as the penalty spot by construction.

### Scoring against labels

Each method tracked all 1041 consecutive frames of the right camera, and the track was
then checked at the 53 hand-reviewed frames (47 of which contain the ball). A hit is a
tracked centre within 80 px of the labelled centre. This is real accuracy, not a proxy.

| Method | Recall | Centre error | s / frame | Full sweeps | Candidates rejected |
|---|---|---|---|---|---|
| physics | **0.979** (46 / 47) | 2.6 px | **0.197** | 517 | 25 |
| velocity | **0.979** | 2.7 px | 0.198 | 522 | 30 |
| sweep | **0.979** | 2.7 px | 0.291 | 1041 | 0 |
| kalman | 0.213 | 2.2 px | 0.286 | 885 | 752 |

## What worked

- **Searching where the ball should be beats searching everywhere.** Physics and velocity
  match the exhaustive sweep's recall exactly, with half the full sweeps and 32 % less
  time.
- **Physics ties constant velocity rather than beating it.** At 45 fps the ball moves
  almost linearly between frames, and in a small-sided game it mostly rolls, so a
  parabola adds little. We kept physics because its fitted vertical acceleration
  signals when the ball is in the air, which the 3D stage uses.

## Results

| Camera | Recall on reviewed frames | Centre error | Detector ceiling |
|---|---|---|---|
| Right | **0.979** | 2.6 px | — |
| Left | **0.850** | 1.9 px | 0.875 |

The detector ceiling is how often the fine-tuned model finds the ball when the tile
holding it is searched directly (35 of 40 reviewed left-camera frames). Tracking reaches
85.0 % against a ceiling of 87.5 %, so on the left camera the limit is the detector, not
the tracking.

## What broke, and limits

- **Our own sweep had a coverage bug.** The tile origins never reached the right or
  bottom edge, so about 600 px of width was never searched. The first test found no
  ball in 60 frames, partly because that frame's ball sat at x = 3800 of 3840. Fixed by
  adding a final tile flush with each edge.
- **The Kalman variant diverges.** Its constant-acceleration prediction drifts off the
  ball, the real detection then fails the plausibility gate and is rejected, which
  starves the filter and makes the drift worse. It stays in the code as a documented
  negative result; it would need damping on the acceleration term.
- **The physics model assumes a fixed camera.** A parabola in image space is only
  ballistic if the camera does not move. On the broadcast footage, which pans to follow
  the ball, this assumption is false, so `05-panorama-broadcast.md` uses the sweep
  method there.
- **The score rests on 47 labelled balls** from one camera and one segment.

## Reproduce

```bash
python tracking/eval_ball.py --tracks ../outputs/tracking/final_right/tracks.json \
    --labels ../inference_out/labels_right/all.json
# compare motion models on the same frames:
python tracking/track.py --video "../raw_footage/Right camera (stereo pair).mp4" \
    --player-weights yolo26x.pt --ball-weights ../training_runs/tiled_m/weights/best.pt \
    --tracker configs/trackers/tracktrack_static.yaml --ball-method kalman \
    --out ../outputs/tracking/ball_kalman --start 1940 --frames 1041
```
