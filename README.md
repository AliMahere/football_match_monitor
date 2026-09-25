# Football match monitor

Detection, tracking and 3D positions of players and the ball from fixed pitch-side
cameras, plus tracking on panoramic and broadcast footage.

Start with **[WRITEUP.md](WRITEUP.md)** for what was built, why, and what broke. The
documents in `docs/` hold every trial and number behind it.

## Results

| | Result |
|---|---|
| **Players** (req. 1) | 19 identities per camera over 23 s; identity survived 52 of 68 crossings |
| **Ball** (req. 2) | 97.9 % recall (right camera) and 85.0 % (left) against hand-labelled positions, 2–3 px error |
| **3D** (req. 3) | cameras calibrated and synced from the footage alone; pitch 33.8 × 21.5 m, baseline 17.9 m; ball height by stereo on 57 % of frames |
| **Panoramic and broadcast** (req. 4) | player tracking on both; ball found with stock weights; not validated against labels |

| Document | Covers |
|---|---|
| [01-detection](docs/01-detection.md) | six published detectors tested, labelling, fine-tuning at native scale |
| [02-player-tracking](docs/02-player-tracking.md) | four trackers compared, identity audit on crossings |
| [03-ball-tracking](docs/03-ball-tracking.md) | four ball motion models scored against labels |
| [04-3d-mapping](docs/04-3d-mapping.md) | sync, calibration from the pitch, stereo and single-camera ball height |
| [05-panorama-broadcast](docs/05-panorama-broadcast.md) | moving-camera tracking, where the models stop transferring |

## Layout

```
detection/        pre-annotation, labels, dataset, tiling, fine-tuning, evaluation
tracking/         player and ball tracking, ball motion models, identity audit, CVAT export
mapping3d/        sync, lens, calibration, 3D positions, figures
tools/            camera point-picking app, H.264 conversion, dataset build script
configs/trackers/ tracker settings; ablations/ holds the comparison runs
docs/             one document per stage, plus figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.14 and CUDA on an RTX 5070 Ti (12 GB). YOLO26 weights
download on first use.

The footage is **not** in this repository. Scripts expect it next to the repository, and
write everything they produce outside it:

```
../raw_footage/        the four source videos
../outputs/tracking/   tracker runs, one folder each
../outputs/mapping3d/  calibration, sync and 3D results
```

The 3D stage reads these from `FMM_FOOTAGE`, `FMM_TRACKS` and `FMM_WORK` when set.
Each document ends with the commands that reproduce its numbers.

## Outputs

Sent separately rather than committed, since they are derived from the footage:

- four 30-second annotated clips: left camera, right camera, broadcast, panoramic;
- per-frame tracks (`tracks.json`) and 3D positions (`players.json`, `ball.json`).

## Handling the footage

The match footage is real and shows minors. It stays on the machine it was processed on.
Nothing derived from it, frames or annotated video, is committed here, and labelling was
done in a local CVAT instance. The only images in this repository are plots.
