# 05 — Panoramic and broadcast footage

Code: `tracking/track.py`. Configs: `configs/trackers/tracktrack_moving_tuned.yaml`,
`configs/trackers/ablations/`. Requirement 4.

## Goal

Get tracking working on the panoramic or broadcast sample, and find out where it breaks
and why. The brief asks for one; we ran both, because they fail differently.

Both samples come from the same Pixellot Air system at a different venue from the
stereo pair: a full-size grass pitch, youth players, 25 fps. Broadcast mode is a
virtual pan-and-zoom crop of the panorama, not a separate camera, so its view moves to
follow the ball.

## What we tried

### Resolution

| Sample | Frame | Median player height | Player height after resizing to 1280 |
|---|---|---|---|
| Panoramic | 3840 × 900 | 77 px | **26 px** |
| Broadcast | 1920 × 1080 | 91 px | 61 px |

The panorama's 4.3 : 1 shape is the problem. Squeezed into a square network input,
players shrink to 26 px. Running players at 2560 px instead finds 49 people in a test
frame against 29 at 1280.

### A tracker for a moving camera

The fixed-camera config is wrong here in two independent ways. The view pans and zooms,
so motion must be compensated before association. And the footage is 25 fps rather than
45, so every buffer expressed in frames lasts a different time. Three configurations,
the same 750 broadcast frames:

| Config | Unique IDs | Mean lifetime | Fragments |
|---|---|---|---|
| fixed-camera config, no motion compensation | 67 | 148.7 | 10 |
| motion compensation only | 55 | 186.9 | 2 |
| **motion compensation + tuned** | **45** | **232.1** | **1** |

Tuned values (`tracktrack_moving_tuned.yaml`): `track_buffer` 30 → 90 (3.6 s at 25 fps,
since the camera can look away from a player for seconds), `lost_match_thr` 0 → 0.9
(the relaxed rebinding pass was disabled entirely at 0), and `match_thresh` 0.7 → 0.8
(motion compensation leaves residual error). Compensation and tuning each removed about
18 % of the identities.

### The ball

The fine-tuned ball model from `01-detection.md` does not transfer. Checked by cropping
around every detection and looking:

| Model | Broadcast | Panoramic |
|---|---|---|
| fine-tuned on our pitch, tiled | confident boxes (0.47–0.57) on a crowd, the sky and a banner | no detections at all |
| stock YOLO26 `sports ball` | 4 of 5 checked frames on the real ball (whole frame at 1280 px) | 0.94 and 0.92 on the real ball, 0.36 on a red marker, one unclear (640 px tiles, as in the clip) |

The training ball was white, on artificial turf, seen from one high fixed angle, with 87
examples. Here the ball, grass and viewpoint all differ. Stock YOLO26 was used for the
ball on both samples. Its confidence is also meaningful here, real balls score around
0.9 and junk around 0.3, so a 0.4 threshold was applied.

## What worked

| Sample | Players | Ball | Tracker | Ball search |
|---|---|---|---|---|
| Broadcast | stock YOLO26, 1280 px | stock YOLO26, 640 px tiles | tuned moving-camera config | sweep |
| Panoramic | stock YOLO26, 2560 px | stock YOLO26, 640 px tiles | fixed-camera config | physics |

Broadcast uses the sweep method because the physics model's parabola is only ballistic
when the camera is still. The panorama is fixed, so physics applies.

## Results

The 30-second submission clips:

| Sample | Frames | Unique IDs | Frames with a ball detection |
|---|---|---|---|
| Broadcast | 750 | 45 | 240 (32 %) |
| Panoramic | 750 | 73 | 352 (47 %) |

**The ball figures are detection counts, not accuracy.** Neither sample has labelled
frames, so these say how often the detector reported a ball, not how often it was right.
They were spot-checked by eye on individual frames, not measured.

## What broke, and limits

- **We reported broadcast ball numbers that measured the wrong thing, and withdrew
  them.** With no labels, we counted frames where any ball-class box appeared and
  treated that as recall. With about 80 tiles searched per frame, some box almost always
  appears. That is how the fine-tuned model looked like it was finding the ball in 95 % of
  frames while actually boxing crowds and banners. The same error made the physics model
  look worse than the sweep method on broadcast, when it may have been correctly
  rejecting the clutter. All of those figures were withdrawn; only the visual checks
  above are kept. On the stereo cameras the ball was always scored against labelled
  positions, and those numbers stand.
- **Coaches and spectators are tracked as players.** Nothing separates the playing area
  from the touchline here, unlike the caged stereo pitch. That inflates the panorama to
  73 identities and gives the coach in the broadcast foreground an ID. A pitch mask
  would fix it; it is not in these clips.
- **Players leaving the broadcast view get new IDs.** The camera follows the ball, so
  players leave and re-enter frame constantly, and nothing links a returning player to
  their old identity. Much of the 45 comes from that, not from tracker failures.
- **Nothing here is validated against labels.** A 40-frame broadcast labelling set was
  prepared but not labelled.

## Reproduce

```bash
python tracking/track.py \
    --video "../raw_footage/Broadcast mode (separate sample, the view follows the ball).mp4" \
    --player-weights yolo26x.pt --ball-weights yolo26x.pt --ball-class 32 \
    --tracker configs/trackers/tracktrack_moving_tuned.yaml \
    --ball-tile 640 --ball-imgsz 640 --ball-conf 0.4 --ball-method sweep \
    --out ../outputs/tracking/submit_broadcast --start 3000 --frames 750 --video-out
python tracking/track.py \
    --video "../raw_footage/Panoramic camera (separate wide-angle sample).mp4" \
    --player-weights yolo26x.pt --ball-weights yolo26x.pt --ball-class 32 --imgsz 2560 \
    --tracker configs/trackers/tracktrack_static.yaml \
    --ball-tile 640 --ball-imgsz 640 --ball-conf 0.4 --ball-method physics \
    --out ../outputs/tracking/submit_panorama --start 6000 --frames 750 \
    --video-out --video-width 3840
```
