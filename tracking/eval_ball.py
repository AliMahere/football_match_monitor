"""Score a ball track against the hand-reviewed frames.

The reviewed frames are 20 apart, so they cannot test continuity directly. What
they can do is check, at 53 known instants, whether the track was on the ball.
Running the tracker over every frame in between and then sampling at those
instants is therefore a real accuracy measure rather than a proxy.

A prediction counts as correct if its centre is within --tolerance pixels of the
labelled centre, which at 4K is a couple of ball widths.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def centre(bbox_xywh):
    x, y, w, h = bbox_xywh
    return np.array([x + w / 2, y + h / 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=80.0)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    names = {c["id"]: c["name"] for c in labels["categories"]}
    truth = {}
    for image in labels["images"]:
        index = int(Path(image["file_name"]).stem.split("_")[-1])
        truth[index] = None
    for annotation in labels["annotations"]:
        image = next(i for i in labels["images"] if i["id"] == annotation["image_id"])
        if names[annotation["category_id"]] != "ball":
            continue
        index = int(Path(image["file_name"]).stem.split("_")[-1])
        truth[index] = centre(annotation["bbox"])

    data = json.loads(args.tracks.read_text())
    predicted = {r["frame"]: r["ball"] for r in data["frames"]}

    hits = misses = false_alarms = correct_absences = 0
    errors = []
    for index, target in sorted(truth.items()):
        if index not in predicted:
            continue
        box = predicted[index]
        got = None if box is None else np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        if target is None:
            if got is None:
                correct_absences += 1
            else:
                false_alarms += 1
            continue
        if got is None:
            misses += 1
            continue
        distance = float(np.linalg.norm(got - target))
        if distance <= args.tolerance:
            hits += 1
            errors.append(distance)
        else:
            false_alarms += 1
            misses += 1

    present = hits + misses
    summary = {
        "method": data["summary"].get("ball_method"),
        "frames_scored": sum(1 for i in truth if i in predicted),
        "gt_ball_present": present,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_absences": correct_absences,
        "recall": round(hits / present, 3) if present else None,
        "mean_centre_error_px": round(float(np.mean(errors)), 1) if errors else None,
        "seconds_per_frame": data["summary"].get("seconds_per_frame"),
        "sweeps": data["summary"].get("ball_sweeps"),
        "rejected": data["summary"].get("ball_candidates_rejected"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
