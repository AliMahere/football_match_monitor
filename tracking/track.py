"""Track players and the ball on a fixed-camera clip.

Players and the ball need different treatment. Players are large and survive a
plain downscaled pass, so they go through the detector once per frame. The ball
is about 34 px at 4K and disappears when the frame is resized, so it is searched
for at native resolution inside a window around where the track predicts it to
be, falling back to a full tiled sweep when the track is lost.

Outputs a tracks.json holding every frame's boxes and identities, and optionally
an annotated video.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import subprocess

import cv2
import numpy as np
from ultralytics import YOLO

from ball_methods import TRACKERS

BALL, PLAYER = 0, 1


class BallSearcher:
    """Finds the ball at native resolution, guided by a motion model.

    The motion model decides where to look and which candidates to believe; see
    ball_methods.py for the alternatives being compared.
    """

    def __init__(self, model, conf, window, tile, overlap, lost_after, motion, imgsz=None, ball_class=BALL):
        self.model = model
        self.conf = conf
        # Kept separate from the crop size on purpose. The ball model was trained
        # at one scale, so on footage where the ball is smaller the crop has to be
        # upscaled to put it back at that scale.
        self.imgsz = imgsz or tile
        self.ball_class = ball_class
        self.window = window
        self.tile = tile
        self.overlap = overlap
        self.lost_after = lost_after
        self.motion = motion
        self.sweeps = 0
        self.searches = 0
        self.rejected = 0

    def _detect(self, crop):
        result = self.model.predict(
            crop, imgsz=self.imgsz, conf=self.conf, classes=[self.ball_class], verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(scores))
        return boxes[best], float(scores[best])

    @staticmethod
    def _origins(total, tile, overlap):
        """Tile starts that actually reach the far edge, so nothing is left unsearched."""
        step = max(1, tile - overlap)
        starts = list(range(0, max(1, total - tile + 1), step))
        if starts[-1] + tile < total:
            starts.append(max(0, total - tile))
        return starts

    def _sweep(self, frame):
        """Full pass over overlapping native-resolution tiles."""
        self.sweeps += 1
        height, width = frame.shape[:2]
        best = None
        for top in self._origins(height, self.tile, self.overlap):
            for left in self._origins(width, self.tile, self.overlap):
                found = self._detect(frame[top : top + self.tile, left : left + self.tile])
                if found is None:
                    continue
                box, score = found
                box = box + [left, top, left, top]
                if best is None or score > best[1]:
                    best = (box, score)
        return best

    @staticmethod
    def _centre(box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def update(self, frame):
        height, width = frame.shape[:2]
        predicted = self.motion.predict()

        if predicted is not None:
            self.searches += 1
            # Widen the search the longer the ball has been missing.
            size = int(self.window * (1 + self.motion.misses))
            size = int(min(size, min(height, width)))
            left = int(np.clip(predicted[0] - size / 2, 0, width - size))
            top = int(np.clip(predicted[1] - size / 2, 0, height - size))
            found = self._detect(frame[top : top + size, left : left + size])
            if found is not None:
                box = found[0] + [left, top, left, top]
                if self.motion.plausible(self._centre(box)):
                    self.motion.update(self._centre(box))
                    return box, found[1], "window"
                self.rejected += 1

        found = self._sweep(frame)
        if found is not None:
            box = found[0]
            if self.motion.plausible(self._centre(box)):
                self.motion.update(self._centre(box))
                return box, found[1], "sweep"
            self.rejected += 1

        self.motion.update(None)
        if self.motion.misses > self.lost_after:
            self.motion.reset()
        return None, 0.0, "lost"



class H264Writer:
    """Pipe frames to ffmpeg.

    OpenCV's pip wheels carry no H.264 encoder, so VideoWriter quietly falls back
    to MPEG-4 Part 2. OpenCV reads that back without complaint, which hides the
    problem until someone tries to play the file in a browser or QuickTime. The
    pixel format and the relocated index below are what those players actually
    need; the codec name alone is not enough.
    """

    def __init__(self, path, size, fps):
        import imageio_ffmpeg

        width, height = size
        self.size = size
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        self.process.wait()


def annotate(frame, players, ball):
    for x1, y1, x2, y2, track_id in players:
        colour = ((37 * int(track_id)) % 255, (17 * int(track_id)) % 255, 200)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
        cv2.putText(
            frame, f"#{int(track_id)}", (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3, cv2.LINE_AA,
        )
    if ball is not None:
        x1, y1, x2, y2 = ball
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 28, (0, 0, 255), 4)
        cv2.putText(
            frame, "ball", (cx + 32, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--player-weights",
        required=True,
        type=Path,
        help="run on the whole frame; the tile-trained model must not be used here",
    )
    parser.add_argument("--player-class", type=int, default=0)
    parser.add_argument(
        "--ball-weights", required=True, type=Path, help="run on native-resolution crops"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ball-conf", type=float, default=0.25)
    parser.add_argument("--ball-window", type=int, default=768)
    parser.add_argument("--ball-tile", type=int, default=1280)
    parser.add_argument(
        "--ball-imgsz",
        type=int,
        default=None,
        help="network input for ball crops; larger than --ball-tile upscales a small ball",
    )
    parser.add_argument("--ball-lost-after", type=int, default=5)
    parser.add_argument("--ball-method", default="physics", choices=list(TRACKERS))
    parser.add_argument(
        "--ball-class",
        type=int,
        default=BALL,
        help="class index in the ball model; COCO sports ball is 32",
    )
    parser.add_argument("--no-ball", action="store_true")
    parser.add_argument("--video-out", action="store_true")
    parser.add_argument("--video-width", type=int, default=1920)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Two models on purpose. The tile-trained checkpoint only ever saw players at
    # native scale, so on a downscaled full frame it finds about half of them; the
    # stock weights match the labelled counts exactly there. The tile-trained one is
    # much better on the ball, which is only ever searched for at native resolution.
    model = YOLO(str(args.player_weights))
    ball_model = YOLO(str(args.ball_weights))
    searcher = BallSearcher(
        ball_model, args.ball_conf, args.ball_window, args.ball_tile,
        args.ball_tile // 4, args.ball_lost_after, TRACKERS[args.ball_method](),
        imgsz=args.ball_imgsz, ball_class=args.ball_class,
    )

    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    fps = capture.get(cv2.CAP_PROP_FPS)

    writer = None
    records = []
    started = time.perf_counter()
    lifetimes = defaultdict(int)
    ball_modes = defaultdict(int)

    for offset in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break
        index = args.start + offset

        result = model.track(
            frame, imgsz=args.imgsz, conf=args.conf, classes=[args.player_class],
            tracker=args.tracker, persist=True, verbose=False,
        )[0]

        players = []
        if result.boxes is not None and result.boxes.id is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(xyxy, ids):
                players.append([*box.tolist(), int(track_id)])
                lifetimes[int(track_id)] += 1

        ball_box = None
        if not args.no_ball:
            ball_box, ball_score, mode = searcher.update(frame)
            ball_modes[mode] += 1
            if ball_box is not None:
                ball_box = [float(v) for v in ball_box]

        records.append({"frame": index, "players": players, "ball": ball_box})

        if args.video_out:
            if writer is None:
                height, width = frame.shape[:2]
                size = (args.video_width, int(height * args.video_width / width))
                # H.264 must be even-dimensioned.
                size = (size[0] // 2 * 2, size[1] // 2 * 2)
                writer = H264Writer(args.out / "tracked.mp4", size, fps)
            drawn = annotate(frame.copy(), players, ball_box)
            writer.write(cv2.resize(drawn, size))

        if offset % 100 == 0:
            print(f"frame {index}: {len(players)} players, ball={'yes' if ball_box else 'no'}")

    capture.release()
    if writer is not None:
        writer.release()

    per_frame = [len(r["players"]) for r in records]
    fragments = sum(1 for v in lifetimes.values() if v < 10)
    summary = {
        "video": args.video.name,
        "tracker": args.tracker,
        "frames": len(records),
        "unique_ids": len(lifetimes),
        "mean_players_per_frame": round(float(np.mean(per_frame)), 2) if per_frame else 0,
        "mean_track_lifetime": round(float(np.mean(list(lifetimes.values()))), 1) if lifetimes else 0,
        "tracks_under_10_frames": fragments,
        "ball_frames_found": sum(v for k, v in ball_modes.items() if k != "lost"),
        "ball_modes": dict(ball_modes),
        "ball_sweeps": searcher.sweeps,
        "ball_window_searches": searcher.searches,
        "ball_candidates_rejected": searcher.rejected,
        "ball_method": args.ball_method,
        "seconds_per_frame": None,
    }
    summary["seconds_per_frame"] = round((time.perf_counter() - started) / max(1, len(records)), 3)
    (args.out / "tracks.json").write_text(json.dumps({"summary": summary, "frames": records}))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
