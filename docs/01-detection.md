# 01 — Detecting players and the ball

Code: `detection/`. Used by requirements 1 and 2.

## Goal

Find the best detector for players and for the ball **on this footage**, and only train
our own if nothing off the shelf is good enough.

The footage decided a lot. Both stereo cameras are 4K at 45 fps over a small-sided,
caged artificial pitch, filmed from an elevated position with wide, visibly curved
lenses. Players are about 400 px tall. The ball is 19–76 px wide, 34 px on average.

## What we tried

Candidates came from Roboflow Universe, Hugging Face, GitHub and the SoccerNet
literature. Their published scores were high, but almost all were measured on broadcast
television, so they were used only as a shortlist. Six were run on the same six frames
of the left camera.

| Candidate | Trained on | Players / frame | Ball found | Verdict |
|---|---|---|---|---|
| YOLO26x, COCO weights, 1280 px | nothing football-specific | 9–11 | 4 / 6 | tight, confident boxes |
| YOLO26x, COCO weights, tiled at native 4K | — | 10–14 | 5 / 6 | best recall; 6.5 s / frame |
| Roboflow `football-players-detection` v20 (RF-DETR-m) | broadcast | 10–12 | 6 / 6 | ball box on the penalty spot most frames |
| RF-DETR fine-tuned on SoccerNet | broadcast | 9–12 | 6 / 6 | duplicate boxes; yellow bibs read as referees |
| YOLO11m football players (HF) | broadcast | 0–5 | — | published 99 % player AP; unusable here |
| YOLO11n football ball (HF) | broadcast | — | 0 / 6 | zero even tiled at full resolution |
| Roboflow `football-ball-detection` v4 | broadcast | — | 0 / 6 | published mAP 89 %; zero here |

Three purpose-built ball detectors found the ball zero times, including at native
resolution where it is full size. The broadcast domain gap, not resolution, is the
cause.

Getting some candidates to run at all took work: the SoccerNet RF-DETR checkpoint only
loads with the library's deprecated patch-14 class, at a resolution that is a multiple
of 56.

## What worked

### Building labels for this pitch

No ground truth existed, so we made it. The Roboflow player model pre-annotated 100
frames per camera, every 20 frames from the same start in both, so the two cameras stay
paired for the 3D stage. The pre-annotations were corrected by hand in a local CVAT
instance, since the footage cannot be uploaded anywhere.

One systematic error made correction much faster to plan: on the left camera the
penalty spot was detected as a ball in 88 of 100 frames, always at the same pixel,
because the camera never moves.

53 frames per camera (1940–2980) were fully reviewed. Goalkeeper and referee were
folded into `player`, because the stock COCO model can only report people and balls
and a four-class comparison would penalise it for something it cannot do. That gives
**1007 player and 87 ball boxes**, split into 84 training and 22 test images. The test
split is the end of the clip rather than a random sample: frames under half a second
apart are near-duplicates, and a random split would leak between train and test.

### Choosing a base model

Scored on the 53 reviewed frames per camera, two classes:

| Camera | Model | mAP@50 | mAP@50-95 | Precision | Recall | Ball AP | Player AP |
|---|---|---|---|---|---|---|---|
| Left | YOLO26x | **0.734** | 0.487 | **0.949** | 0.913 | **0.528** | 0.940 |
| Left | Roboflow v20 | 0.662 | **0.620** | 0.802 | **0.922** | 0.348 | **0.975** |
| Right | YOLO26x | **0.850** | 0.614 | **0.966** | 0.960 | **0.721** | 0.978 |
| Right | Roboflow v20 | 0.843 | **0.767** | 0.793 | **0.974** | 0.702 | **0.984** |

The two fail in opposite directions. The Roboflow model draws tighter boxes but
invents about 25 % more players than exist. YOLO26 predicts almost exactly the right
count and is better on the ball. For tracking, precision matters more than box
tightness, since every invented box can start a false identity. **We fine-tuned
YOLO26.**

### Fine-tuning at the ball's real scale

A 4K frame resized to a 1280 px input shrinks the 34 px ball to about 11 px, which is at
the limit of what the network can resolve. So training uses **native-resolution tiles**
(1280 px, 256 px overlap) instead of downscaled frames. 15 % of tiles holding nothing
are kept on purpose: the pitch markings mistaken for the ball appear only in empty
tiles, and the model has to see them labelled as background.

Tiled dataset: 457 train tiles (1174 player, 95 ball) and 131 test tiles (284 player, 39
ball). YOLO26m, AdamW at 1e-3, cosine schedule, scale augmentation held to 0.3 so the
ball is not shrunk further. Early stopping ended the run at epoch 50, best at 20.

## Results

Held-out test frames only, both models run with the same tiling so the comparison
isolates fine-tuning:

| Camera | Model | mAP@50 | mAP@50-95 | Ball AP | Player AP |
|---|---|---|---|---|---|
| Left | fine-tuned | **0.792** | **0.594** | **0.703** | 0.881 |
| Left | stock YOLO26 | 0.584 | 0.440 | 0.257 | 0.911 |
| Right | fine-tuned | **0.970** | **0.693** | **1.000** | 0.939 |
| Right | stock YOLO26 | 0.889 | 0.600 | 0.881 | 0.897 |

The gain is almost entirely the ball. Player AP is roughly neutral.

## What broke, and limits

- **Our first evaluation leaked.** It scored the fine-tuned model on all reviewed
  frames, most of which it had trained on. It was redone on the held-out split only;
  the leaked numbers are not used anywhere.
- **The ball numbers rest on few examples.** The held-out set holds 10 balls on the left
  camera and 9 on the right. "1.000" means nine balls were found.
- **The test split is not a generalisation test.** It is the tail of the same clip, under
  half a second from training frames.
- **Validation and test are the same images.** There was not enough data for three
  splits, so early stopping saw the frames the numbers are reported on.
- **Tile training has a side effect.** The model only ever saw players at native scale,
  so on a downscaled full frame it finds about half of them (3 of 7, 8 of 11 where stock
  YOLO26 finds all). Tracking therefore uses stock YOLO26 for players and the fine-tuned
  model only for the ball, inside native-resolution crops.
- **The fine-tuned ball model does not leave this pitch.** On the Pixellot footage it
  boxes crowds, sky and banners, or nothing at all; stock YOLO26 does better there. See
  `05-panorama-broadcast.md`. With 87 ball examples from one venue, this was the stated
  risk, and it happened.

## Reproduce

```bash
python detection/infer_frames.py --video "../raw_footage/Left camera (stereo pair).mp4" \
    --model roboflow-players --out ../inference_out/cvat_left_3zvbc \
    --frames 100 --start 1000 --stride 20
# repeat for the right camera, correct both in a local CVAT, and save each COCO 1.0
# export in its folder as "cvat left.json" / "cvat right.json"; then:
bash tools/generate_dataset.sh     # reviewed frames only, from frame 1940 on
python detection/tile_dataset.py --src ../datasets/ours --out ../datasets/ours_tiled
python detection/finetune_yolo26.py --data ../datasets/ours_tiled/data.yaml --name tiled_m
python detection/eval_detector.py --labels ../inference_out/labels_left/test.json \
    --frames ../inference_out/cvat_left_3zvbc/frames \
    --model ../training_runs/tiled_m/weights/best.pt --slice --log ../inference_out/eval.log
```
