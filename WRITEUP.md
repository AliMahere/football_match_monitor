# Writeup

## What we built

A pipeline that takes the two fixed cameras from pixels to metres, and a version of its
tracking adapted to Pixellot footage.

| Requirement | Result | Detail |
|---|---|---|
| 1. Players with stable IDs | 19 identities for about 9 players on screen over 23 s per camera. Identity survived **52 of 68 crossings** when audited by eye | [02](docs/02-player-tracking.md) |
| 2. Ball | **97.9 % recall** on the right camera, 85.0 % on the left, 2–3 px centre error, scored against hand-labelled positions | [03](docs/03-ball-tracking.md) |
| 3. 3D mapping | Both cameras calibrated from the pitch alone, synced to −115 frames. Players and ball in metres, ball height by stereo on 57 % of frames | [04](docs/04-3d-mapping.md) |
| 4. Panoramic and broadcast | Player tracking on both, with a moving-camera tracker config for broadcast. Ball detection works with stock weights; not validated with labels | [05](docs/05-panorama-broadcast.md) |

How the detector was chosen and trained is in [01](docs/01-detection.md).

## What we used, and why

**Stock YOLO26 for players, our fine-tuned YOLO26 for the ball.** Six published football
detectors were run on our footage. Every broadcast-trained one did worse than generic
COCO weights, and three dedicated ball detectors found the ball zero times, one of them
advertising 89 % mAP. The footage is an elevated view of a caged pitch, far from the
broadcast images those models learnt. So we labelled our own frames, pre-annotating with
a model and correcting in a local CVAT.

**Training and searching at the ball's real size.** A 4K frame resized for the network
shrinks the 34 px ball to 11 px. We trained on native-resolution tiles and search for
the ball in native-resolution crops. Ball AP on held-out frames went from 0.26 to 0.70
on the harder camera. The tile-trained model then finds only half the players on a
resized full frame, which is why players use the stock model.

**TrackTrack, without appearance features.** It halved the identity count of ByteTrack
and removed fragments. Appearance ReID made every metric worse: both teams wear plain
bibs, so teammates look identical to an embedding.

**A motion model for the ball.** Predicting where the ball will be and searching there
matches an exhaustive search on recall at 32 % less time.

**Calibration from the pitch itself, not a pitch calibrator.** TVCalib and PnLCalib
assume a 105 × 68 m field. This one turned out to be 33.8 × 21.5 m. Its dimensions are
fitted together with both cameras, with metric scale from the 3 × 2 m goals.

**Sync by geometry.** The stereo files have no audio, no usable timestamps, and clip
lengths that differ only because of pre-roll frames. Once both cameras share one
calibration, the same player's feet land in the same place from both views only at the
right lag, which gives one sharp answer, stable across the clip.

## What broke

The most useful lessons came from things that failed.

- **Published scores predicted nothing about this footage.** The model with the best
  published player accuracy found almost no players here.
- **The same model is right in one place and wrong in another.** The fine-tuned ball
  model scores 0.979 recall on our pitch, boxes crowds and banners on the Pixellot
  broadcast, and finds nothing on the panorama. 87 examples from one venue was the stated
  risk.
- **An assumption hidden in a motion model.** Fitting a parabola to image positions
  assumes the camera is still. It was built for the fixed cameras and does not hold on a
  view that follows the ball.
- **Summary metrics hid identity errors.** 19 IDs and zero fragments coexisted with 11
  real identity failures on one camera. We only found them by building an audit: detect
  the risky moments, render each as a filmstrip, and judge them.
- **Measuring the wrong thing.** Twice we reported numbers that counted something other
  than what we meant. The first detector evaluation included training frames. On the
  broadcast footage, with no labels, we counted "a ball box appeared" as "the ball was
  found". Both were caught and withdrawn, and the second one reversed a conclusion. Where
  labels existed (the stereo ball, the detector), numbers were checked against them;
  where they did not, the writeup says so.
- **Our own bugs.** A tile sweep that never reached the frame edge, a Kalman filter that
  diverges, and a lens inverse that diverged in the image corners and moved the pitch
  width by a metre once fixed. OpenCV's wheels also write video most players cannot open,
  which a readback test in OpenCV did not catch.
- **The first idea about the cameras was wrong.** We expected them side by side with a
  narrow overlap. Calibration put them 17.9 m apart with stereo over most of the pitch.

## With more time

- **Labels where there are none.** Label the prepared broadcast set and some panorama
  frames, so requirement 4 gets measured numbers instead of spot checks. Label identities
  on a consecutive segment, so tracking gets HOTA and IDF1 rather than an audit rate.
- **A pitch mask.** It removes coaches and spectators on the Pixellot footage and
  off-pitch false balls everywhere. The calibration already defines the playing area.
- **Ball detection that generalises.** Train on several venues, or combine a temporal
  detector such as WASB with a still-camera check, so a static mark cannot pass as a
  ball.
- **Keep identity through occlusion.** Every lost identity during a crossing happened
  because the hidden player stopped being detected. Detecting partly hidden players, or
  holding their tracks longer, targets that directly.
- **Motion-compensated ball physics.** The tracker already estimates camera motion on
  broadcast; applying it to the ball before fitting the parabola would make the physics
  model valid there.
- **3D over the whole clip, and fused.** Calibration and sync already cover the full five
  minutes; only the tracked 20.6 s is mapped. Fitting whole ball flights between bounces,
  using stereo and single-camera observations together, and confirming the goal size
  would pin down scale and height further.
- **Speed.** The clips ran at 3.7–5.0 frames per second on an RTX 5070 Ti. Live operation
  needs 45 fps per camera, and the stereo pair needs two cameras at once, so the pipeline
  is roughly 20× too slow. In order of expected payoff:
  - **TensorRT.** Export both YOLO26 models to TensorRT engines in FP16, then try INT8
    with a calibration set drawn from this footage. Accuracy has to be re-checked after
    each step with the existing evaluation scripts, since quantisation hits small
    objects like the ball first.
  - **Batch the tiles.** The ball search runs the network once per tile, one after
    another. Sending all tiles of a frame as one batch uses the GPU far better.
  - **Stop re-reading the frame on the CPU.** 4K decoding, resizing and cropping all run
    on the CPU today. GPU decoding (NVDEC) and GPU-side crops keep frames on the card.
  - **Overlap the stages.** Decoding, player detection, ball search and tracking run in
    strict sequence. Running them as a pipeline, and both cameras in parallel, hides most
    of the waiting.
  - **Search less.** Most tiles hold only grass. Sweep only near the predicted ball and
    inside the pitch mask, and fall back to a full sweep rarely.
- **Software design.** The code is organised by stage and each stage is a command-line
  script that hands JSON files to the next. That suited experiments; a product needs
  more:
  - **An installable package.** Seven files find their neighbours by editing
    `sys.path`. A proper package with a `pyproject.toml` removes that and makes the code
    importable from anywhere.
  - **Typed data between stages.** Detections and tracks travel as bare lists such as
    `[x1, y1, x2, y2, id]`. Small dataclasses for detections, tracks and camera
    calibration would make the interfaces explicit and catch mismatches early.
  - **One configuration per camera setup.** The broadcast run takes thirteen
    command-line flags. A config file per setup (fixed stereo, panoramic, broadcast),
    holding the model choices, tile sizes, thresholds and tracker, would replace them and
    record exactly how each result was produced.
  - **Clear interfaces for the parts that vary.** Detector, tracker and ball motion model
    should each sit behind one interface, as the ball motion models already do, so a
    model can be swapped without touching the pipeline.
  - **Streaming instead of files.** Stages currently finish before the next one starts.
    A frame-by-frame pipeline is what a live match monitor needs, and it is also what the
    speed work above depends on.
  - **Tests in the repository.** The end-to-end check run before submission reproduced
    every stage, and the calibration to 0.000 %. It belongs in the repo as an automated
    regression test, with unit tests for the geometry and a CI job, so later changes
    cannot silently move the numbers.
  - **Logging instead of print statements**, so long runs can be monitored and compared.
