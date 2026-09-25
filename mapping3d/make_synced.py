"""
Step 1b - write the two videos out IN SYNC, using the lag measured by sync.py.

    python mapping3d/make_synced.py [--preview-only]

    work/synced/left_synced.mp4    \\  same start instant, same length:
    work/synced/right_synced.mp4   /   frame n of one = frame n of the other
    work/synced/side_by_side.mp4   1920x540 preview with both frame numbers burned in

With lag < 0 (right = left + lag) the right camera started later, so the first |lag|
frames of the LEFT video are dropped; both are then cut to the common length.
Re-encoded (frame-accurate cuts are impossible with stream copy) on the GPU with NVENC
when available, else libx264.
"""
import argparse
import json
import os
import subprocess
from paths import FOOTAGE, WORK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "synced")
SRC = {s: os.path.join(FOOTAGE, f"{s.capitalize()} camera (stereo pair).mp4")
       for s in ("left", "right")}


def readable_frames(path):
    """Frames with pts >= 0 (the header count also includes skipped pre-roll frames)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "packet=pts", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return sum(int(x) >= 0 for x in out.split() if x.strip())


def encoder():
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                         text=True).stdout
    if "h264_nvenc" in enc:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lag = json.load(open(os.path.join(WORK, "sync", "sync.json")))["lag_frames"]
    n = {s: readable_frames(p) for s, p in SRC.items()}
    start = {"left": max(0, -lag), "right": max(0, lag)}
    length = min(n["left"] - start["left"], n["right"] - start["right"])
    info = {"lag_frames": lag, "readable_frames": n, "first_frame_used": start,
            "synced_length_frames": length, "synced_length_s": length / 45.0}
    json.dump(info, open(os.path.join(OUT, "synced.json"), "w"), indent=2)
    print(info)
    enc = encoder()
    trim = {s: f"trim=start_frame={start[s]}:end_frame={start[s] + length},setpts=PTS-STARTPTS"
            for s in SRC}
    if not a.preview_only:
        for s in SRC:
            subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-i", SRC[s],
                            "-vf", trim[s], *enc, "-pix_fmt", "yuv420p", "-an",
                            os.path.join(OUT, f"{s}_synced.mp4")], check=True)
    label = ("drawtext=text='{side}  frame %{{eif\\:n+{off}\\:d}}':x=20:y=20:fontsize=36:"
             "fontcolor=yellow:box=1:boxcolor=black@0.6")
    fc = (f"[0:v]{trim['left']},scale=960:540,{label.format(side='LEFT', off=start['left'])}[l];"
          f"[1:v]{trim['right']},scale=960:540,{label.format(side='RIGHT', off=start['right'])}[r];"
          "[l][r]hstack=inputs=2[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-i", SRC["left"], "-i", SRC["right"],
                    "-filter_complex", fc, "-map", "[v]", *enc, "-pix_fmt", "yuv420p",
                    os.path.join(OUT, "side_by_side.mp4")], check=True)


if __name__ == "__main__":
    main()
