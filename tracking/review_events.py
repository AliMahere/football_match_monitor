"""Turn identity-risk events into one picture each, so a tracker can be scored by eye.

The question "did the same person keep the same id" cannot be answered from
summary counts. It needs someone to look. This makes looking cheap: every event
becomes a single filmstrip showing the same players before, during and after the
moment, with their ids drawn on. A verdict per strip, and the score follows.

It also hunts the failure that matters most and is invisible to the event list:
a track dying and the same player picking up a NEW id moments later. Those show
up as a vanish followed by a fresh track starting nearby, and are flagged as
handover candidates.

Writes a filmstrip per event plus verdicts.csv, ready to fill in.

Example:
    python review_events.py --tracks final_right/tracks.json \
        --video "../raw_footage/Right camera (stereo pair).mp4" --out review_right
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

COLOURS = [(60, 220, 60), (60, 60, 240), (240, 180, 40), (220, 60, 220), (40, 220, 220)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = (x2 - x1) * (y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def collect(frames):
    by_track, by_frame = defaultdict(dict), {}
    for record in frames:
        by_frame[record["frame"]] = record["players"]
        for box in record["players"]:
            by_track[int(box[4])][record["frame"]] = box[:4]
    return by_track, by_frame


def find_events(frames, by_track, min_iou, gap, edge, handover_window, handover_distance):
    events = []

    active = {}
    for record in frames:
        boxes = record["players"]
        current = set()
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                overlap = iou(boxes[i][:4], boxes[j][:4])
                if overlap < min_iou:
                    continue
                pair = tuple(sorted((int(boxes[i][4]), int(boxes[j][4]))))
                current.add(pair)
                entry = active.setdefault(pair, [record["frame"], record["frame"], 0.0])
                entry[1] = record["frame"]
                entry[2] = max(entry[2], overlap)
        for pair in [p for p in active if p not in current]:
            start, end, peak = active.pop(pair)
            events.append({"type": "crossing", "ids": list(pair), "start": start,
                           "end": end, "peak_iou": round(peak, 3)})
    for pair, (start, end, peak) in active.items():
        events.append({"type": "crossing", "ids": list(pair), "start": start,
                       "end": end, "peak_iou": round(peak, 3)})

    last_frame = frames[-1]["frame"]
    first_frame = frames[0]["frame"]
    starts = {tid: min(f) for tid, f in by_track.items()}
    ends = {tid: max(f) for tid, f in by_track.items()}

    for track_id, positions in by_track.items():
        indices = sorted(positions)
        for previous, nxt in zip(indices, indices[1:]):
            if nxt - previous > gap:
                events.append({"type": "reappear", "ids": [track_id],
                               "start": previous, "end": nxt, "peak_iou": ""})

        end_frame = ends[track_id]
        if end_frame >= last_frame - gap:
            continue
        box = positions[end_frame]
        point = centre(box)
        if not (edge < point[0] < 3840 - edge and edge < point[1] < 2160 - edge):
            continue

        # A new track appearing near where this one died is the classic handover.
        partner = None
        for other, other_start in starts.items():
            if other == track_id or not (end_frame < other_start <= end_frame + handover_window):
                continue
            distance = float(np.linalg.norm(centre(by_track[other][other_start]) - point))
            if distance <= handover_distance and (partner is None or distance < partner[1]):
                partner = (other, distance)

        if partner:
            events.append({"type": "handover", "ids": [track_id, partner[0]],
                           "start": end_frame, "end": starts[partner[0]],
                           "peak_iou": "", "gap_px": round(partner[1], 1)})
        else:
            events.append({"type": "vanish", "ids": [track_id], "start": end_frame,
                           "end": end_frame, "peak_iou": ""})

    order = {"handover": 0, "crossing": 1, "reappear": 2, "vanish": 3}
    events.sort(key=lambda e: (order[e["type"]], -float(e.get("peak_iou") or 0)))
    return events


def filmstrip(capture, event, by_frame, out_path, panel=520):
    start, end = event["start"], event["end"]
    wanted = sorted({start - 12, start, (start + end) // 2, end, end + 12})
    ids = set(event["ids"])

    panels = []
    anchor = []
    for index in wanted:
        if index not in by_frame:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        boxes = [b for b in by_frame[index] if int(b[4]) in ids]
        if not boxes:
            # Neither track exists on this frame, which is the whole point of a
            # handover. Stay where they were rather than jumping across the pitch.
            boxes = anchor
        if not boxes:
            continue
        anchor = boxes

        xs = [v for b in boxes for v in (b[0], b[2])]
        ys = [v for b in boxes for v in (b[1], b[3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.8 + 200
        span = int(max(span, 400))
        left = int(np.clip(cx - span / 2, 0, frame.shape[1] - span))
        top = int(np.clip(cy - span / 2, 0, frame.shape[0] - span))
        crop = frame[top : top + span, left : left + span].copy()

        for box in by_frame[index]:
            track_id = int(box[4])
            if track_id not in ids:
                continue
            colour = COLOURS[sorted(ids).index(track_id) % len(COLOURS)]
            p1 = (int(box[0] - left), int(box[1] - top))
            p2 = (int(box[2] - left), int(box[3] - top))
            cv2.rectangle(crop, p1, p2, colour, 4)
            cv2.putText(crop, f"#{track_id}", (p1[0], max(24, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3, cv2.LINE_AA)

        crop = cv2.resize(crop, (panel, panel))
        cv2.putText(crop, f"f{index}", (8, panel - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(crop)

    if not panels:
        return False
    cv2.imwrite(str(out_path), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--edge", type=int, default=120)
    parser.add_argument("--handover-window", type=int, default=45)
    parser.add_argument("--handover-distance", type=float, default=400.0)
    parser.add_argument("--max-events", type=int, default=40)
    args = parser.parse_args()

    frames = json.loads(args.tracks.read_text())["frames"]
    by_track, by_frame = collect(frames)
    events = find_events(frames, by_track, args.iou, args.gap, args.edge,
                         args.handover_window, args.handover_distance)

    args.out.mkdir(parents=True, exist_ok=True)
    strips = args.out / "strips"
    strips.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))

    rows = []
    for number, event in enumerate(events[: args.max_events]):
        name = f"{number:03d}_{event['type']}_{'-'.join(map(str, event['ids']))}.jpg"
        if not filmstrip(capture, event, by_frame, strips / name):
            continue
        rows.append({
            "event": number, "type": event["type"], "ids": "-".join(map(str, event["ids"])),
            "start": event["start"], "end": event["end"],
            "peak_iou": event.get("peak_iou", ""), "strip": name, "verdict": "",
        })
    capture.release()

    with (args.out / "verdicts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["event", "type", "ids", "start", "end", "peak_iou",
                                 "strip", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for event in events:
        counts[event["type"]] += 1
    print(f"events: {dict(counts)} (total {len(events)})")
    print(f"{len(rows)} filmstrips -> {strips}")
    print(f"fill the verdict column in {args.out / 'verdicts.csv'}")
    print("\nverdict values:")
    print("  ok      identities stayed on the same people")
    print("  swap    two identities traded people")
    print("  split   one person ended up with a new id (handover confirmed)")
    print("  lost    the person was dropped and never recovered")
    print("  na      not a real event, or too unclear to call")


if __name__ == "__main__":
    main()
