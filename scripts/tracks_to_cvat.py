"""Export tracker output as CVAT-importable tracks, so review is correction not labelling.

Importing boxes alone means re-assigning every identity by hand. Importing tracks
means the identities arrive with them, and the work becomes splitting a track
where it swapped and merging two where it fragmented. That is the difference
between hours and minutes, and it is what makes proper MOT metrics affordable.

Two formats:

  mot   MOT 1.1, a zip holding gt/gt.txt and labels.txt. CVAT imports this
        directly, and TrackEval reads the same layout, so the corrected export
        feeds straight into HOTA/IDF1 without conversion.
  cvat  CVAT-for-video 1.1 XML, which carries keyframes and outside flags.

Frame numbers are rebased to the clip, since a CVAT task built from an extracted
segment starts at zero while our records carry absolute indices.

Example:
    python tracks_to_cvat.py --tracks final_right/tracks.json --out cvat_right_tracks --format mot
"""

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def load(tracks_path):
    data = json.loads(tracks_path.read_text())
    frames = data["frames"]
    base = frames[0]["frame"] if frames else 0
    return data, frames, base


def write_mot(frames, base, out, include_ball):
    """MOT 1.1: frame,id,left,top,width,height,conf,class,visibility — 1-based frames."""
    lines = []
    ball_id = 10_000  # kept clear of player ids
    for record in frames:
        index = record["frame"] - base + 1
        for box in record["players"]:
            x1, y1, x2, y2, track_id = box
            lines.append(
                f"{index},{int(track_id)},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},1,1,1"
            )
        if include_ball and record.get("ball"):
            x1, y1, x2, y2 = record["ball"]
            lines.append(
                f"{index},{ball_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},1,2,1"
            )

    out.mkdir(parents=True, exist_ok=True)
    archive = out / "mot_tracks.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("gt/gt.txt", "\n".join(lines) + "\n")
        bundle.writestr("labels.txt", "player\nball\n")
    return archive, len(lines)


def write_cvat_xml(frames, base, out, include_ball, width, height):
    by_track = {}
    for record in frames:
        index = record["frame"] - base
        for x1, y1, x2, y2, track_id in record["players"]:
            by_track.setdefault(("player", int(track_id)), []).append((index, x1, y1, x2, y2))
        if include_ball and record.get("ball"):
            x1, y1, x2, y2 = record["ball"]
            by_track.setdefault(("ball", 0), []).append((index, x1, y1, x2, y2))

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "size").text = str(len(frames))
    ET.SubElement(task, "mode").text = "interpolation"
    labels = ET.SubElement(task, "labels")
    for name in ("player", "ball"):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "attributes")

    for number, ((label_name, track_id), boxes) in enumerate(sorted(by_track.items())):
        track = ET.SubElement(
            root, "track", id=str(number), label=label_name, source="manual"
        )
        boxes.sort()
        present = {b[0] for b in boxes}
        for index, x1, y1, x2, y2 in boxes:
            ET.SubElement(
                track, "box", frame=str(index), outside="0", occluded="0", keyframe="1",
                xtl=f"{x1:.2f}", ytl=f"{y1:.2f}", xbr=f"{x2:.2f}", ybr=f"{y2:.2f}",
            )
        # CVAT needs an explicit closing frame or the track runs to the end.
        last = boxes[-1][0]
        if last + 1 < len(frames) and last + 1 not in present:
            _, x1, y1, x2, y2 = boxes[-1]
            ET.SubElement(
                track, "box", frame=str(last + 1), outside="1", occluded="0", keyframe="1",
                xtl=f"{x1:.2f}", ytl=f"{y1:.2f}", xbr=f"{x2:.2f}", ybr=f"{y2:.2f}",
            )

    out.mkdir(parents=True, exist_ok=True)
    path = out / "annotations.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path, len(by_track)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--format", choices=["mot", "cvat"], default="mot")
    parser.add_argument("--no-ball", action="store_true")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    args = parser.parse_args()

    data, frames, base = load(args.tracks)
    include_ball = not args.no_ball

    if args.format == "mot":
        path, rows = write_mot(frames, base, args.out, include_ball)
        print(f"{rows} rows -> {path}")
        print("CVAT: Upload annotations -> MOT 1.1 -> this zip")
        print("TrackEval reads the same gt/gt.txt layout after you export the corrections")
    else:
        path, count = write_cvat_xml(
            frames, base, args.out, include_ball, args.width, args.height
        )
        print(f"{count} tracks -> {path}")
        print("CVAT: Upload annotations -> CVAT for video 1.1 -> this file")

    print(f"frames {base}..{frames[-1]['frame']} rebased to 0..{len(frames) - 1}")


if __name__ == "__main__":
    main()
