"""Re-encode tracking output to H.264 so it plays anywhere.

OpenCV's pip wheels ship no H.264 encoder, so VideoWriter falls back to MPEG-4
Part 2 (FMP4). OpenCV reads that back happily, which hides the problem, but
browsers, QuickTime and several desktop players refuse it.

This transcodes to H.264 with yuv420p and a moved index, which is what those
players actually need. ffmpeg comes from the imageio-ffmpeg wheel, so nothing
has to be installed system-wide.

Example:
    python scripts/to_h264.py --inputs ../outputs/tracking/*/tracked.mp4
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def convert(source, destination, crf, preset):
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-loglevel", "error",
        "-i", str(source),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        # Players are far pickier about pixel format than about the codec name.
        "-pix_fmt", "yuv420p",
        # Put the index at the front so the file starts without a full download.
        "-movflags", "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--suffix", default="_h264")
    parser.add_argument("--crf", type=int, default=23, help="lower is better quality")
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()

    for source in args.inputs:
        if not source.exists():
            print(f"missing: {source}")
            continue
        destination = source.with_name(f"{source.stem}{args.suffix}.mp4")
        convert(source, destination, args.crf, args.preset)
        before = source.stat().st_size / 1e6
        after = destination.stat().st_size / 1e6
        print(f"{source}  {before:.0f}MB -> {destination.name}  {after:.0f}MB")


if __name__ == "__main__":
    main()
