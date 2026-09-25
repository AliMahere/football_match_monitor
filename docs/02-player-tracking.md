# 02 — Tracking players with stable identities

Code: `tracking/track.py`, `tracking/review_events.py`, `tracking/score_review.py`.
Configs: `configs/trackers/`. Requirement 1.

## Goal

Keep one identity per player across the clip, including through occlusion and players
crossing paths.

## What we tried

Detection comes from stock YOLO26x on the whole frame (see `01-detection.md` for why
not the fine-tuned model). Global motion compensation is switched off in every config:
it exists for moving cameras, and these are bolted down, so it could only cost time and
add error.

Four association methods, right camera, the same 1041 frames (23 s, about 9.3 players on
screen):

| Tracker | Unique IDs | Mean lifetime (frames) | Tracks under 10 frames | s / frame |
|---|---|---|---|---|
| **TrackTrack** | **19** | **501.8** | **0** | 0.048 |
| ByteTrack | 38 | 254.9 | 17 | 0.047 |
| BoT-SORT | 39 | 248.3 | 19 | 0.047 |
| BoT-SORT + appearance ReID | 53 | 183.5 | 33 | 0.063 |

**Appearance ReID made tracking worse, not better.** It is the usual advice for
surviving crossings, and it produced the most identities and the most fragments here.
Both teams wear plain coloured bibs, so teammates are nearly identical to an appearance
embedding; ReID supplies confident wrong matches and damages association that motion
alone was handling. Telling these players apart would need jersey numbers, not generic
embeddings.

## What worked

TrackTrack with motion compensation off (`configs/trackers/tracktrack_static.yaml`). On
the final 1041-frame runs:

| Camera | Unique IDs | Mean lifetime | Fragments |
|---|---|---|---|
| Right | 19 | 501.8 | 0 |
| Left | 19 | 494.1 | 2 |

### Checking identity, not just counting it

Those numbers cannot show a swap: if two players trade identities, the ID count and
track lifetimes do not change at all. Ground-truth track IDs do not exist, and
labelling them for HOTA or IDF1 would take hours. So identity was audited where it is
actually at risk.

`review_events.py` finds the risky moments automatically:

- **crossing** — two player boxes overlap (IoU above 0.15);
- **handover** — a track ends in open play and a new ID starts within 1 s and 400 px,
  the signature of one person being renamed;
- **reappear** — a track goes missing and comes back;
- **vanish** — a track ends in open play with no successor.

Each event becomes one filmstrip showing the same players before, during and after. A
person marks each strip `ok`, `swap`, `split` or `lost` (per identity when two are
involved), and `score_review.py` turns the verdicts into rates.

## Results

| | Left | Right | Both |
|---|---|---|---|
| Crossings survived | 24 / 33 (72.7 %) | 28 / 35 (80.0 %) | **52 / 68 (76.5 %)** |
| Handovers that were real ID switches | 3 / 3 | 4 / 4 | **7 / 7** |
| Reappears that kept the right ID | 2 / 2 | — | 2 / 2 |
| Identities that survived their event | 35 / 47 (74.5 %) | 34 / 45 (75.6 %) | **69 / 92 (75.0 %)** |

- **Identity survived 52 of 68 crossings.** One crossing in five loses an identity. The
  tracker is good, not solved.
- **The handover heuristic was right every time**, 7 of 7 across both cameras. It is a
  way to find identity switches without any labels.
- **Every failure during a crossing was a loss, not a swap.** When one player hides
  another, the detector returns one box instead of two, and the hidden player's track
  starves and ends. That points the fix at detection under occlusion and at the
  track buffer, not at the association cost.
- **Reappears always recovered the right person**, so recovery after a short gap works.

## What broke, and limits

- **These are audit rates, not HOTA or IDF1.** They cover the events the detector found;
  a swap during an overlap below the IoU threshold is not counted.
- **The summary metrics flattered the tracker.** 19 IDs and zero fragments on the right
  camera coexist with 11 real identity failures. Only looking found them.
- **The clips are short.** 23 s per camera. Failures per minute should not be quoted
  from so small a sample.
- For proper MOT metrics, `tracking/tracks_to_cvat.py` exports tracks with IDs as MOT
  1.1, which CVAT imports for correction and TrackEval reads unchanged.

## Reproduce

```bash
python tracking/track.py --video "../raw_footage/Right camera (stereo pair).mp4" \
    --player-weights yolo26x.pt --ball-weights ../training_runs/tiled_m/weights/best.pt \
    --tracker configs/trackers/tracktrack_static.yaml \
    --out ../outputs/tracking/final_right --start 1940 --frames 1041 --video-out
python tracking/review_events.py --tracks ../outputs/tracking/final_right/tracks.json \
    --video "../raw_footage/Right camera (stereo pair).mp4" --out ../outputs/tracking/review_right
# fill the verdict column in review_right/verdicts.csv, then:
python tracking/score_review.py --verdicts ../outputs/tracking/review_right/verdicts.csv \
    --tracks ../outputs/tracking/final_right/tracks.json
```
