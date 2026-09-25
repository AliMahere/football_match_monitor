"""
STEP 1 — Do the two cameras stand in the same spot, or apart?  (PyQt6 app)

Install (once):
    pip install PyQt6 opencv-python-headless numpy

    Use opencv-python-HEADLESS. The normal "opencv-python" package ships its
    own copy of Qt, which can clash with PyQt and make windows misbehave.
    If you already have it:  pip uninstall opencv-python

Run:
    python check_cameras_qt.py

All instructions are inside the app (the "Instructions" tab).
"""
import json
import os
import subprocess
import sys
from itertools import combinations

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QImage, QKeySequence, QPainter,
                         QPen, QPixmap, QShortcut)
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                             QFileDialog, QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsItemGroup, QGraphicsLineItem,
                             QGraphicsRectItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView,
                             QHBoxLayout, QHeaderView, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QSpinBox, QSplitter,
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QTextBrowser, QToolBar, QVBoxLayout, QWidget)

# The 3D stage owns the output paths; share them rather than guessing a folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mapping3d"))
from paths import WORK  # noqa: E402

TEAL = QColor("#1D9E75")
CORAL = QColor("#D85A30")
YELLOW = QColor("#F2C200")
WHITE = QColor("#FFFFFF")
KIND_COLOR = {"ground": TEAL, "high": CORAL}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MIN_GROUND, GOOD_GROUND = 5, 6   # 4 would always fit exactly (0 px) and prove nothing
MIN_HIGH, GOOD_HIGH = 2, 4
GROUND_OK = 5.0   # correctly clicked ground points miss by less than this (px)


# ----------------------------------------------------------------------------
# The maths (no user interface here)
# ----------------------------------------------------------------------------
def readable_frames(path, header_count):
    """Frames OpenCV can actually return. These clips were cut from longer recordings and
    start with "pre-roll" frames (negative timestamps) that players and OpenCV skip, but
    the header count includes them (left: 175 of them, right: 54). Counted with ffprobe;
    falls back to the header count if ffprobe isn't installed."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "packet=pts", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout
        pts = [int(x) for x in out.split() if x.strip().lstrip("-").isdigit()]
        return sum(p >= 0 for p in pts) or header_count
    except (OSError, subprocess.SubprocessError):
        return header_count


def measured_offset():
    """The lag measured by mapping3d/sync.py (right frame = left frame + lag), or None."""
    path = os.path.join(WORK, "sync", "sync.json")
    try:
        return int(json.load(open(path))["lag_frames"])
    except (OSError, ValueError, KeyError):
        return None


class Video:
    """A video (or still image) kept open, so stepping and playing are fast."""

    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path) if path.lower().endswith(IMAGE_EXT) else None
        if self.image is not None:
            self.cap, self.total, self.fps = None, 1, 0.0
        else:
            self.cap = cv2.VideoCapture(path)
            self.total = readable_frames(path, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.next = 0      # the frame a plain read() returns; no seek needed then

    def read(self, n):
        """Frame n, or None if it doesn't exist. A still image is every frame."""
        if self.image is not None:
            return self.image
        if not 0 <= n < self.total:
            return None
        if n != self.next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, img = self.cap.read()
        self.next = n + 1 if ok else -1
        return img if ok else None


def _flatness(a, b, c):
    """0 = three points on one line; ~0.43 = a perfect triangle."""
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    longest = max(np.sum((a - b) ** 2), np.sum((b - c) ** 2), np.sum((a - c) ** 2))
    return area / longest if longest > 0 else 0.0


def analyse(pairs, image_width=None):
    """Learn one warp from GROUND points, test it on HIGH points.

    Ground points always fit one warp (the grass is flat), cameras apart or not.
    High points fit that same warp ONLY if the cameras stand in the same spot.
    """
    index = [i for i, p in enumerate(pairs) if p["left"] and p["right"]]
    kinds = [pairs[i]["kind"] for i in index]
    L = np.float32([pairs[i]["left"] for i in index]).reshape(-1, 2)
    R = np.float32([pairs[i]["right"] for i in index]).reshape(-1, 2)
    ground = np.array([k == "ground" for k in kinds], dtype=bool)
    high = ~ground

    if ground.sum() < MIN_GROUND:
        raise ValueError(f"You need at least {MIN_GROUND} finished GROUND pairs "
                         f"(you have {int(ground.sum())}). {GOOD_GROUND} or more is better.")
    if high.sum() < MIN_HIGH:
        raise ValueError(f"You need at least {MIN_HIGH} finished HIGH pairs "
                         f"(you have {int(high.sum())}). {GOOD_HIGH} or more is better.")
    # Ground points must be spread out, not all on one line (e.g. the halfway line).
    # A warp learned from one line can't predict anything off that line.
    spread = np.linalg.svd(L[ground] - L[ground].mean(axis=0), compute_uv=False)
    if spread[1] < 0.15 * spread[0]:
        raise ValueError(
            "Your ground points lie almost on one straight line (for example, all on "
            "the halfway line). A warp learned from one line can't predict anything "
            "away from it.\n\nAdd ground spots AWAY from that line, for example:\n"
            "  - a corner of a white box\n"
            "  - where a seam between two barrier panels meets the grass\n"
            "  - the feet of a player standing still")

    # A warp needs 4 ground points with no 3 of them on one line. So "4 points on
    # the halfway line + 1 elsewhere" is NOT enough: the warp can still tilt freely.
    best = 0.0
    Lg = L[ground]
    for quad in combinations(range(len(Lg)), 4):
        best = max(best, min(_flatness(*Lg[list(t)]) for t in combinations(quad, 3)))
    if best < 0.05:
        raise ValueError(
            "Most of your ground points are on one line, with only one spot away "
            "from it. That is not enough: the warp can still tilt freely and will "
            "predict nonsense.\n\nYou need at least TWO ground spots away from that "
            "line, far from each other. Add at least one more.")

    # Method 0 = use every point. No outlier removal: the misses ARE the evidence.
    H, _ = cv2.findHomography(L[ground], R[ground], 0)
    if H is None:
        raise ValueError("Could not fit the warp. Spread the ground points out more.")
    predicted = cv2.perspectiveTransform(L.reshape(-1, 1, 2), H).reshape(-1, 2)
    miss = np.linalg.norm(predicted - R, axis=1)
    g_med = float(np.median(miss[ground]))
    h_med = float(np.median(miss[high]))

    # If the ground points disagree, find which ones: RANSAC keeps the largest
    # group of ground points that agree within GROUND_OK px; the rest are suspects.
    suspects = []
    gi = np.flatnonzero(ground)
    if g_med > GROUND_OK and len(gi) >= 6:
        _, mask = cv2.findHomography(L[ground], R[ground], cv2.RANSAC, GROUND_OK)
        if mask is not None and mask.sum() >= 5:
            suspects = [index[j] for j, ok in zip(gi, mask.ravel()) if not ok]

    if image_width and h_med > 0.5 * image_width:
        verdict = "broken"         # no real camera setup can cause misses this big
    elif g_med > GROUND_OK:
        verdict = "check"          # ground points disagree: the test can't be trusted
    elif h_med <= max(4.0, 2 * g_med):
        verdict = "same"
    elif h_med >= max(10.0, 4 * g_med):
        verdict = "apart"
    else:
        verdict = "unclear"
    return {"index": index, "kinds": kinds, "right": R, "predicted": predicted,
            "miss": miss, "ground_median": g_med, "high_median": h_med,
            "verdict": verdict, "suspects": suspects}


def save_result_picture(img_right, result, path):
    vis = img_right.copy()
    bgr = {"ground": (117, 158, 29), "high": (48, 90, 216)}
    for n, (k, r, p, m) in enumerate(zip(result["kinds"], result["right"],
                                          result["predicted"], result["miss"])):
        r = tuple(int(v) for v in r)
        p = tuple(int(v) for v in p)
        cv2.line(vis, r, p, (0, 194, 242), 2)
        cv2.circle(vis, r, 7, bgr[k], -1)
        cv2.circle(vis, p, 9, (0, 194, 242), 2)
        label = f"{result['index'][n] + 1}: {m:.0f}px"
        cv2.putText(vis, label, (r[0] + 10, r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, bgr[k], 2)
    cv2.imwrite(path, vis)


# ----------------------------------------------------------------------------
# Image view: zoom with the wheel, pan with right/middle drag, left-click to mark
# ----------------------------------------------------------------------------
class ImageView(QGraphicsView):
    clicked = pyqtSignal(float, float)

    def __init__(self, placeholder):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.pixmap_item = None
        self.marks = []
        self._pan = None
        self.show_placeholder(placeholder)

    def show_placeholder(self, text):
        self.scene().clear()
        self.pixmap_item, self.marks = None, []
        item = self.scene().addSimpleText(text)
        item.setBrush(QColor("#bbbbbb"))
        font = item.font()
        font.setPointSize(14)
        item.setFont(font)

    def set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        if self.pixmap_item and self.sceneRect() == QRectF(0, 0, w, h):
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))   # keep the zoom
            return
        self.scene().clear()
        self.marks = []
        self.pixmap_item = self.scene().addPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        QTimer.singleShot(0, self.fit)

    def fit(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.pixmap_item:
            f = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(f, f)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton and self.pixmap_item:
            p = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(p):
                self.clicked.emit(p.x(), p.y())

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = event.position()
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - int(d.x()))
            v.setValue(v.value() - int(d.y()))

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # --- drawing marks ----------------------------------------------------
    def clear_marks(self):
        for item in self.marks:
            self.scene().removeItem(item)
        self.marks = []

    def add_marker(self, x, y, label, color, selected=False, dashed=False):
        group = QGraphicsItemGroup()
        r = 11 if selected else 7
        pen = QPen(WHITE if selected else color, 3 if selected else 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        ring = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        ring.setPen(pen)
        group.addToGroup(ring)
        if not dashed:  # crosshair with a gap, so the exact spot stays visible
            for x1, y1, x2, y2 in ((-r - 5, 0, -2, 0), (2, 0, r + 5, 0),
                                   (0, -r - 5, 0, -2), (0, 2, 0, r + 5)):
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(QPen(color, 1.5))
                group.addToGroup(line)
        if label:
            text = QGraphicsSimpleTextItem(label)
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            text.setFont(font)
            text.setBrush(QBrush(color))
            tx, ty = (r + 4, r + 2) if dashed else (r + 4, -r - 18)  # misses go below
            text.setPos(tx, ty)
            br = text.boundingRect()
            bg = QGraphicsRectItem(tx - 2, ty - 1, br.width() + 4, br.height() + 2)
            bg.setBrush(QColor(0, 0, 0, 170))
            bg.setPen(QPen(Qt.PenStyle.NoPen))
            group.addToGroup(bg)
            group.addToGroup(text)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        group.setZValue(10)
        group.setPos(x, y)
        self.scene().addItem(group)
        self.marks.append(group)

    def add_line(self, x1, y1, x2, y2, color):
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        line = self.scene().addLine(x1, y1, x2, y2, pen)
        line.setZValue(5)
        self.marks.append(line)


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
INSTRUCTIONS = f"""
<h3>What this tool finds out</h3>
<p>Whether your two cameras stand in the <b>same spot</b> (only turned different
ways) or <b>apart</b>. Only cameras that stand apart can measure depth together,
like two eyes.</p>

<h3>How to use it</h3>
<ol>
<li>Click <b>Open LEFT camera</b> and <b>Open RIGHT camera</b> at the top and pick
the videos (still images also work).</li>
<li><b>Sync the videos</b> (the <b>Sync</b> tab). The cameras didn't start
recording at the same moment, so the same frame number shows two different
moments. Line them up <i>before</i> you click any points — see the Sync tab.</li>
<li>Type a frame number and press <b>Load frame</b>. Pick a moment where the
centre circle is visible in both cameras.</li>
<li>Keep <span style="color:{TEAL.name()}"><b>Ground</b></span> selected. Click a
spot on the LEFT image, then the <b>exact same spot</b> on the RIGHT image.
Repeat for about {GOOD_GROUND} spots.<br>
<i>Good ground spots:</i> places where two lines cross — where the halfway line
meets a touchline, where the halfway line crosses the centre circle — and the
centre spot. Each must be one exact spot you can find in both images.<br>
<b>Spread them out</b> — not all on one line. The halfway line alone is not
enough: also use a corner of a white box, or where a seam between two barrier
panels meets the grass.<br>
<i>Avoid:</i> the "edge" of the circle (it's a different spot in each camera),
and moving players.</li>
<li>Press <span style="color:{CORAL.name()}"><b>High</b></span> (or key H). Click
about {GOOD_HIGH} high spots the same way.<br>
<i>Good high spots:</i> fence tops, poles, a player's head, a treetop —
anything that is not on the grass. Tall and far-away spots help most.</li>
<li>Press <b>Run check</b>. The answer appears in the Result tab.</li>
</ol>

<h3>Tips for accurate clicks</h3>
<ul>
<li><b>Mouse wheel</b> zooms in and out. <b>Right-click and drag</b> moves the
image. Press <b>F</b> to see the whole image again.</li>
<li>The <span style="color:{YELLOW.name()}"><b>yellow frame</b></span> shows which
image to click next.</li>
<li>Spread the ground points out. Don't put them all on one line.</li>
<li>Prefer things that don't move. If you use a player, choose a frame where
they stand still.</li>
<li>Mistake? Press <b>U</b> to undo the last click, or select a row in the
Points tab and press <b>Delete</b>.</li>
<li>Press <b>Ctrl+S</b> to save your points so you don't have to click again.</li>
</ul>

<h3>Keyboard shortcuts</h3>
<p>G = ground &nbsp; H = high &nbsp; U = undo &nbsp; R = run check &nbsp;
F = fit images &nbsp; Ctrl+S = save points &nbsp; Delete = remove selected row</p>
<p>Syncing: A / D = both videos 1 frame back / forward &nbsp; Shift+A / Shift+D =
10 frames &nbsp; [ / ] = move only the RIGHT video 1 frame &nbsp; P = play / pause</p>

<h3>How to read the answer</h3>
<ul>
<li><b>SAME SPOT</b>: stereo cannot help. Go to Step 2.</li>
<li><b>APART</b>: stereo is possible in the overlap. Keep it as a check for the
ball later, then go to Step 2.</li>
<li><b>UNCLEAR</b>: add more high points (tall, far ones), or run the check again
after Step 2 fixes the lens bending.</li>
</ul>
"""

SYNC_HELP = """
<p>The two cameras didn't start recording at the same moment. So <b>frame 1000</b>
in the left video and <b>frame 1000</b> in the right video are different moments.
Players move between them, and every point on a player is then wrong.</p>
<p>The <b>offset</b> fixes that: the RIGHT video shows frame
<i>left frame + offset</i>. A negative offset means the right camera started later.</p>
<ol>
<li>Press <b>Use measured offset</b>. It loads the offset measured automatically by
<code>mapping3d/sync.py</code> (−115 frames for this match: the right camera started 2.56 s
later), and is applied by itself when you open both videos. The steps below are for
checking it by eye, or for syncing other footage by hand.</li>
<li>Find a moment that is easy to see in <b>both</b> images, in the overlap (the
centre circle area): the ball is kicked, the ball bounces, a player's foot lands
on a line, a player changes direction.</li>
<li>Step <b>both videos</b> until the LEFT image shows that exact moment.</li>
<li>Now move <b>only the right video</b> until it shows the same moment.
Zoom in (mouse wheel) to compare foot and ball positions.</li>
<li>Check it: step both videos forward and <b>Play</b> a few seconds. The same
things should happen at the same time in both images. Try one more moment
further into the video.</li>
</ol>
<p>The offset is saved with your points.</p>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step 1 — Do the two cameras stand in the same spot?")
        self.resize(1500, 880)
        self.sources = {"left": None, "right": None}
        self.videos = {"left": None, "right": None}
        self.images = {"left": None, "right": None}
        self.shown = {"left": None, "right": None}   # frame number on screen
        self.pairs = []            # each: {"kind", "left": [x, y], "right": [x, y] or None}
        self.mode = "ground"
        self.result = None
        self.selected = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(0)
        self.play_timer.timeout.connect(self.play_tick)

        self._build_toolbar()
        self._build_body()
        self._build_shortcuts()
        self.statusBar().showMessage("Start with the Instructions tab on the left.")
        self.refresh()

    # --- layout -------------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction("Open LEFT camera…", lambda: self.open_source("left"))
        tb.addAction("Open RIGHT camera…", lambda: self.open_source("right"))
        tb.addSeparator()
        tb.addWidget(QLabel("  Left frame: "))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 10_000_000)
        self.frame_spin.setValue(1000)
        self.frame_spin.setMinimumWidth(100)
        tb.addWidget(self.frame_spin)
        self.frame_total_label = QLabel("  ")
        self.frame_total_label.setMinimumWidth(110)
        tb.addWidget(self.frame_total_label)
        tb.addWidget(QLabel("  Right offset: "))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-10_000_000, 10_000_000)
        self.offset_spin.setMinimumWidth(80)
        self.offset_spin.setToolTip("RIGHT video shows frame (left frame + offset). "
                                    "Set it in the Sync tab.")
        tb.addWidget(self.offset_spin)
        tb.addAction("Load frame", self.load_frame)
        tb.addSeparator()
        tb.addAction("Fit images (F)", self.fit_views)
        tb.addSeparator()
        tb.addAction("Save points…", self.save_points)
        tb.addAction("Load points…", self.load_points)

    def _build_body(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Sidebar
        side = QWidget()
        side.setMinimumWidth(380)
        side.setMaximumWidth(520)
        sl = QVBoxLayout(side)

        self.next_label = QLabel()
        self.next_label.setWordWrap(True)
        self.next_label.setStyleSheet(
            "QLabel { background:#2b2b2b; color:white; padding:10px; "
            "border-radius:8px; font-size:15px; }")
        sl.addWidget(self.next_label)

        mode_row = QHBoxLayout()
        self.btn_ground = QPushButton("Ground point (G)")
        self.btn_high = QPushButton("High point (H)")
        for btn, name, color in ((self.btn_ground, "ground", TEAL),
                                 (self.btn_high, "high", CORAL)):
            btn.setCheckable(True)
            btn.setMinimumHeight(38)
            btn.setStyleSheet(
                f"QPushButton {{ border:2px solid {color.name()}; border-radius:6px; "
                f"color:{color.name()}; font-weight:bold; }}"
                f"QPushButton:checked {{ background:{color.name()}; color:white; }}")
            btn.clicked.connect(lambda _, n=name: self.set_mode(n))
            mode_row.addWidget(btn)
        group = QButtonGroup(self)
        group.addButton(self.btn_ground)
        group.addButton(self.btn_high)
        self.btn_ground.setChecked(True)
        sl.addLayout(mode_row)

        self.count_label = QLabel()
        self.count_label.setWordWrap(True)
        sl.addWidget(self.count_label)

        self.tabs = QTabWidget()
        # Tab 1: instructions
        help_box = QTextBrowser()
        help_box.setHtml(INSTRUCTIONS)
        self.tabs.addTab(help_box, "Instructions")
        # Tab 2: sync
        sync = QWidget()
        syl = QVBoxLayout(sync)
        self.sync_label = QLabel()
        self.sync_label.setWordWrap(True)
        self.sync_label.setStyleSheet(
            "QLabel { background:#2b2b2b; color:white; padding:8px; border-radius:6px; }")
        syl.addWidget(self.sync_label)
        for title, buttons in (
                ("Both videos:", (("−1 s", lambda: self.step(-self.second())),
                                  ("−10", lambda: self.step(-10)), ("−1", lambda: self.step(-1)),
                                  ("+1", lambda: self.step(1)), ("+10", lambda: self.step(10)),
                                  ("+1 s", lambda: self.step(self.second())))),
                ("Only the RIGHT video (changes the offset):",
                 (("−10", lambda: self.nudge(-10)), ("−1", lambda: self.nudge(-1)),
                  ("+1", lambda: self.nudge(1)), ("+10", lambda: self.nudge(10))))):
            syl.addWidget(QLabel(f"<b>{title}</b>"))
            row = QHBoxLayout()
            for text, fn in buttons:
                b = QPushButton(text)
                b.setMinimumWidth(40)
                b.clicked.connect(fn)
                row.addWidget(b)
            syl.addLayout(row)
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play (P)")
        self.play_btn.setToolTip("Plays both videos together (slower than real time).")
        self.play_btn.clicked.connect(self.toggle_play)
        guess = QPushButton("Use measured offset")
        guess.setToolTip("The lag measured by mapping3d/sync.py from players seen by both cameras.")
        guess.clicked.connect(self.use_measured_offset)
        row.addWidget(self.play_btn)
        row.addWidget(guess)
        syl.addLayout(row)
        sync_help = QTextBrowser()
        sync_help.setHtml(SYNC_HELP)
        syl.addWidget(sync_help, stretch=1)
        self.tabs.addTab(sync, "Sync")
        # Tab 3: points
        pts = QWidget()
        pl = QVBoxLayout(pts)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Type", "Left (x, y)", "Right (x, y)", "Miss"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in (0, 1, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c in (2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        pl.addWidget(self.table)
        row = QHBoxLayout()
        for text, fn in (("Undo (U)", self.undo), ("Delete selected", self.delete_selected),
                         ("Clear all", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        pl.addLayout(row)
        self.tabs.addTab(pts, "Points")
        # Tab 4: result
        self.result_box = QTextBrowser()
        self.tabs.addTab(self.result_box, "Result")
        sl.addWidget(self.tabs, stretch=1)

        self.run_btn = QPushButton("Run check (R)")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.setStyleSheet("QPushButton { font-size:15px; font-weight:bold; }")
        self.run_btn.clicked.connect(self.run_check)
        sl.addWidget(self.run_btn)
        splitter.addWidget(side)

        # Two image views
        views = QWidget()
        vl = QHBoxLayout(views)
        vl.setContentsMargins(4, 4, 4, 4)
        self.views, self.view_titles = {}, {}
        for s in ("left", "right"):
            col = QVBoxLayout()
            title = QLabel()
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setMinimumHeight(28)
            view = ImageView(f"Open the {s.upper()} camera file\n(button at the top)")
            view.clicked.connect(lambda x, y, s=s: self.on_click(s, x, y))
            col.addWidget(title)
            col.addWidget(view, stretch=1)
            vl.addLayout(col)
            self.views[s], self.view_titles[s] = view, title
        splitter.addWidget(views)
        splitter.setStretchFactor(1, 1)

    def _build_shortcuts(self):
        for key, fn in (("G", lambda: self.set_mode("ground")),
                        ("H", lambda: self.set_mode("high")),
                        ("U", self.undo), ("R", self.run_check), ("F", self.fit_views),
                        ("Ctrl+S", self.save_points),
                        ("A", lambda: self.step(-1)), ("D", lambda: self.step(1)),
                        ("Shift+A", lambda: self.step(-10)), ("Shift+D", lambda: self.step(10)),
                        ("[", lambda: self.nudge(-1)), ("]", lambda: self.nudge(1)),
                        ("P", self.toggle_play)):
            QShortcut(QKeySequence(key), self, activated=fn)
        delete = QShortcut(QKeySequence("Delete"), self.table, activated=self.delete_selected)
        delete.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

    # --- loading ------------------------------------------------------------
    def confirm_discard(self, why):
        if not self.pairs:
            return True
        answer = QMessageBox.question(
            self, "Clear your points?",
            f"{why} Your clicked points belong to the current images and will be cleared.\n\n"
            "Continue? (Tip: press Cancel, then Save points first.)")
        if answer == QMessageBox.StandardButton.Yes:
            self.pairs, self.result, self.selected = [], None, None
            return True
        return False

    def open_source(self, side):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Open {side.upper()} camera video or image", "",
            "Video or image (*.mp4 *.avi *.mov *.mkv *.m4v *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
            "All files (*)")
        if not path or not self.confirm_discard("You are changing a camera file."):
            return
        self.stop_play()
        self.sources[side] = path
        self.videos[side] = Video(path)
        self.load_side(side)
        if all(self.videos.values()):
            self.tabs.setCurrentIndex(1)
            lag = measured_offset()
            if lag is not None and self.offset_spin.value() == 0 and not self.pairs:
                self.use_measured_offset()
            else:
                self.statusBar().showMessage(
                    "Both videos are open. Sync them first (Sync tab).", 8000)
        self.refresh()

    def frame_of(self, side, frame=None, offset=None):
        frame = self.frame_spin.value() if frame is None else frame
        offset = self.offset_spin.value() if offset is None else offset
        return frame + offset if side == "right" else frame

    def load_side(self, side):
        n = self.frame_of(side)
        img = self.videos[side].read(n)
        if img is None:
            QMessageBox.warning(self, "Could not read",
                                f"Could not read frame {n} from:\n{self.sources[side]}\n\n"
                                f"It has {self.videos[side].total} frames. Try another "
                                "frame number or offset.")
            return False
        self.images[side], self.shown[side] = img, n
        self.views[side].set_image(img)
        return True

    def show_moment(self, frame, offset):
        """Show left frame `frame` and right frame `frame + offset`. All or nothing."""
        frames = {}
        for s in ("left", "right"):
            if self.videos[s]:
                frames[s] = self.videos[s].read(self.frame_of(s, frame, offset))
                if frames[s] is None:
                    return False
        for s, img in frames.items():
            self.images[s], self.shown[s] = img, self.frame_of(s, frame, offset)
            self.views[s].set_image(img)
        self.frame_spin.setValue(frame)
        self.offset_spin.setValue(offset)
        return True

    def goto(self, frame, offset):
        if not any(self.videos.values()):
            self.flash("Open the camera files first (buttons at the top).")
            return
        if not self.confirm_discard("You are moving to a different moment."):
            return
        if not self.show_moment(frame, offset):
            self.flash(f"Left frame {frame} / right frame {frame + offset} doesn't exist "
                       "in both videos.")
        self.refresh()

    def load_frame(self):
        self.stop_play()
        self.goto(self.frame_spin.value(), self.offset_spin.value())

    # --- syncing --------------------------------------------------------------
    def current(self):
        """(left frame, offset) that is on screen now."""
        left, right = self.shown["left"], self.shown["right"]
        frame = left if left is not None else self.frame_spin.value()
        offset = right - left if left is not None and right is not None \
            else self.offset_spin.value()
        return frame, offset

    def second(self):
        fps = [v.fps for v in self.videos.values() if v and v.fps > 0]
        return int(round(fps[0])) if fps else 25

    def step(self, n):
        """Move both videos together."""
        self.stop_play()
        frame, offset = self.current()
        self.goto(frame + n, offset)

    def nudge(self, n):
        """Move only the right video, i.e. change the offset."""
        self.stop_play()
        frame, offset = self.current()
        self.goto(frame, offset + n)

    def use_measured_offset(self):
        lag = measured_offset()
        if lag is None:
            self.flash("No measured offset yet. Run:  python mapping3d/sync.py  (see README).")
            return
        self.stop_play()
        frame, _ = self.current()
        self.goto(max(frame, -lag), lag)
        self.statusBar().showMessage(
            f"Measured offset applied: right frame = left frame {lag:+d}. "
            "Check it on a clear moment (ball kick) if you like.", 8000)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.stop_play()
            return
        if not any(self.videos.values()) or not self.confirm_discard("Playing moves the videos."):
            return
        self.play_timer.start()
        self.play_btn.setText("⏸ Pause (P)")

    def stop_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶ Play (P)")
            self.refresh()

    def play_tick(self):
        frame, offset = self.current()
        if not self.show_moment(frame + 1, offset):
            self.stop_play()
            self.flash("End of the video.")
            return
        self.update_texts()

    def fit_views(self):
        for v in self.views.values():
            v.fit()

    # --- clicking -----------------------------------------------------------
    def expected_side(self):
        return "right" if self.pairs and self.pairs[-1]["right"] is None else "left"

    def on_click(self, side, x, y):
        if self.images["left"] is None or self.images["right"] is None:
            self.flash("Open BOTH camera files first.")
            return
        expect = self.expected_side()
        if side != expect:
            self.flash(f"Click the {expect.upper()} image now (it has the yellow frame).")
            return
        if expect == "left":
            self.pairs.append({"kind": self.mode, "left": [x, y], "right": None})
        else:
            self.pairs[-1]["right"] = [x, y]
        self.result = None
        self.refresh()

    def set_mode(self, mode):
        self.mode = mode
        (self.btn_ground if mode == "ground" else self.btn_high).setChecked(True)
        if self.pairs and self.pairs[-1]["right"] is None:
            self.pairs[-1]["kind"] = mode       # also applies to the point in progress
        self.result = None
        self.refresh()

    def undo(self):
        if not self.pairs:
            return
        if self.pairs[-1]["right"] is not None:
            self.pairs[-1]["right"] = None
        else:
            self.pairs.pop()
        self.result, self.selected = None, None
        self.refresh()

    def delete_selected(self):
        if self.selected is None or self.selected >= len(self.pairs):
            self.flash("Select a row in the Points tab first.")
            return
        self.pairs.pop(self.selected)
        self.result, self.selected = None, None
        self.refresh()

    def clear_all(self):
        if self.pairs and QMessageBox.question(
                self, "Clear all?", "Remove all clicked points?") == QMessageBox.StandardButton.Yes:
            self.pairs, self.result, self.selected = [], None, None
            self.refresh()

    def on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        self.selected = rows[0].row() if rows else None
        self.draw_marks()

    # --- checking -----------------------------------------------------------
    def run_check(self):
        try:
            self.result = analyse(self.pairs, self.images["right"].shape[1])
        except ValueError as e:
            QMessageBox.information(self, "Not ready yet", str(e))
            return
        picture = os.path.join(os.getcwd(), "step1_result.png")
        try:
            save_result_picture(self.images["right"], self.result, picture)
        except Exception:
            picture = None
        self.show_result(picture)
        self.refresh()
        self.tabs.setCurrentWidget(self.result_box)

    def show_result(self, picture):
        r = self.result
        head = {"same": ("SAME SPOT", "#1D9E75"),
                "apart": ("APART", "#534AB7"),
                "unclear": ("UNCLEAR", "#BA7517"),
                "check": ("CHECK YOUR POINTS FIRST", "#A32D2D"),
                "broken": ("THE WARP IS BROKEN", "#A32D2D")}[r["verdict"]]
        if r["suspects"]:
            nums = ", ".join(f"#{i + 1}" for i in r["suspects"])
            fix = (f"<p><b>Suspicious ground points: {nums}</b> (red in the Points tab). "
                   "They disagree with the other ground points. Zoom in and re-click them, "
                   "or delete them, then run the check again.</p>")
        else:
            fix = ("<p>No single point stands out, so all ground points are off together. "
                   "Likely causes:</p><ul>"
                   "<li><b>Not the same physical spot.</b> For example, the 'edge' of the "
                   "centre circle is a different spot in each camera. Use crossings instead: "
                   "where lines meet, the centre spot, where the halfway line crosses the "
                   "circle.</li>"
                   "<li><b>Moving players.</b> If the two videos aren't perfectly in sync, "
                   "feet move between the images. Check the Sync tab, or use only lines "
                   "and fixed objects.</li>"
                   "<li><b>Lens bending.</b> The overlap is at the image edges, where the lens "
                   "bends most. If the clicks are right, do Step 2 first, then repeat Step 1."
                   "</li></ul>")
        body = {
            "same": "The high points fit the grass warp too. The cameras only turn; "
                    "they don't stand apart, so they cannot measure depth together."
                    "<p><b>Next:</b> skip stereo and go to Step 2 (fix lens bending).</p>",
            "apart": "The high points miss the grass warp by a lot, so the cameras stand "
                     "in different places. Stereo is possible in the overlap."
                     "<p><b>Next:</b> keep stereo as a later check for ball height, "
                     "and go to Step 2 (fix lens bending).</p>",
            "unclear": "The high points miss a little, but not clearly more than your "
                       "clicking accuracy."
                       "<p><b>Next:</b> add more high points (tall and far ones help most), "
                       "or run this again after Step 2 fixes the lens bending.</p>",
            "broken": f"The high points miss by {r['high_median']:.0f} px, more than half "
                      "the image width. Real cameras can't cause that: the warp itself is "
                      "wrong. Usually the ground points are too few or too close to one "
                      "line. Add ground spots spread out in different directions, then "
                      "run the check again.",
            "check": f"Your ground points don't agree with each other. They miss by "
                     f"{r['ground_median']:.0f} px, but correctly clicked ground points miss by "
                     f"less than about {GROUND_OK:.0f} px. Until this is fixed, the answer "
                     f"can't be trusted." + fix}[r["verdict"]]
        html = (f"<h2 style='color:{head[1]}'>{head[0]}</h2><p>{body}</p>"
                f"<p>Ground points, typical miss: <b>{r['ground_median']:.1f} px</b> "
                f"(your clicking accuracy)<br>"
                f"High points, typical miss: <b>{r['high_median']:.1f} px</b> (the test)</p>"
                "<p>On the RIGHT image, <span style='color:#F2C200'><b>yellow dashed rings</b></span> "
                "show where the grass warp expected each point. Long yellow lines on "
                "<span style='color:#D85A30'><b>high</b></span> points mean the cameras stand apart. "
                "The Points tab lists every miss.</p>")
        if picture:
            html += f"<p>Result picture saved to:<br><code>{picture}</code></p>"
        self.result_box.setHtml(html)

    # --- save / load --------------------------------------------------------
    def save_points(self):
        if not self.pairs:
            self.flash("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save points", "step1_points.json",
                                              "Points (*.json)")
        if not path:
            return
        frame, offset = self.current()
        with open(path, "w") as f:
            json.dump({"left_source": self.sources["left"],
                       "right_source": self.sources["right"],
                       "frame": frame,             # left frame
                       "right_offset": offset,     # right frame = frame + right_offset
                       "pairs": self.pairs}, f, indent=2)
        self.statusBar().showMessage(f"Saved {len(self.pairs)} points to {path}", 5000)

    def load_points(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load points", "", "Points (*.json)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self.stop_play()
        self.frame_spin.setValue(int(data.get("frame", 0)))
        self.offset_spin.setValue(int(data.get("right_offset", 0)))   # old files: unsynced
        for s in ("left", "right"):
            src = data.get(f"{s}_source")
            if not (src and os.path.exists(src)):
                QMessageBox.information(
                    self, "Find the video",
                    f"The {s.upper()} camera file was not found:\n{src}\n\n"
                    "Please show me where it is now.")
                src, _ = QFileDialog.getOpenFileName(
                    self, f"Find the {s.upper()} camera file", "",
                    "Video or image (*.mp4 *.avi *.mov *.mkv *.m4v *.png *.jpg *.jpeg "
                    "*.bmp *.tif *.tiff);;All files (*)")
            if src:
                self.sources[s] = src
                self.videos[s] = Video(src)
                self.load_side(s)
        self.pairs = data.get("pairs", [])
        self.result, self.selected = None, None
        self.refresh()

    # --- screen update -------------------------------------------------------
    def flash(self, message):
        QApplication.beep()
        self.statusBar().showMessage(message, 5000)

    def refresh(self):
        self.draw_marks()
        self.fill_table()
        self.update_texts()

    def draw_marks(self):
        for s in ("left", "right"):
            self.views[s].clear_marks()
        for i, p in enumerate(self.pairs):
            color = KIND_COLOR[p["kind"]]
            for s in ("left", "right"):
                if p[s]:
                    self.views[s].add_marker(p[s][0], p[s][1], str(i + 1), color,
                                             selected=(i == self.selected))
        if self.result:
            view = self.views["right"]
            for k, r, pr, m in zip(self.result["kinds"], self.result["right"],
                                   self.result["predicted"], self.result["miss"]):
                view.add_line(r[0], r[1], pr[0], pr[1], YELLOW)
                label = f"{m:.0f}px" if k == "high" else ""   # all misses: Points tab
                view.add_marker(pr[0], pr[1], label, YELLOW, dashed=True)

    def fill_table(self):
        miss = {}
        if self.result:
            miss = dict(zip(self.result["index"], self.result["miss"]))
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.pairs))
        fmt = lambda pt: f"{pt[0]:.1f}, {pt[1]:.1f}" if pt else "← click RIGHT image"
        for i, p in enumerate(self.pairs):
            cells = [str(i + 1), p["kind"].capitalize(), fmt(p["left"]), fmt(p["right"]),
                     f"{miss[i]:.1f} px" if i in miss else ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 1:
                    item.setForeground(QBrush(KIND_COLOR[p["kind"]]))
                if self.result and i in self.result["suspects"]:
                    item.setBackground(QBrush(QColor("#F7C1C1")))
                self.table.setItem(i, c, item)
        if self.selected is not None and self.selected < len(self.pairs):
            self.table.selectRow(self.selected)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)

    def update_texts(self):
        done = [p for p in self.pairs if p["right"]]
        n_ground = sum(p["kind"] == "ground" for p in done)
        n_high = sum(p["kind"] == "high" for p in done)
        tick = lambda n, lo, hi: "✓" if n >= hi else ("ok" if n >= lo else f"need {lo}")
        self.count_label.setText(
            f"<span style='color:{TEAL.name()}'><b>Ground pairs: {n_ground}</b></span> "
            f"({tick(n_ground, MIN_GROUND, GOOD_GROUND)}, {GOOD_GROUND}+ is best) &nbsp;&nbsp; "
            f"<span style='color:{CORAL.name()}'><b>High pairs: {n_high}</b></span> "
            f"({tick(n_high, MIN_HIGH, GOOD_HIGH)}, {GOOD_HIGH}+ is best)")
        self.run_btn.setEnabled(n_ground >= MIN_GROUND and n_high >= MIN_HIGH)

        # Left frames that exist in both videos with this offset
        _, offset = self.current()
        lo, hi = 0, None
        for s in ("left", "right"):
            v = self.videos[s]
            if v and v.total > 1:
                shift = offset if s == "right" else 0
                lo = max(lo, -shift)
                hi = v.total - 1 - shift if hi is None else min(hi, v.total - 1 - shift)
        self.frame_total_label.setText(f" ({lo}–{hi})  " if hi is not None else "  ")

        fps = self.second()
        if self.videos["left"] and self.videos["right"]:
            late = "RIGHT" if offset < 0 else "LEFT"
            self.sync_label.setText(
                f"Left frame <b>{self.shown['left']}</b> ↔ right frame "
                f"<b>{self.shown['right']}</b><br>Offset <b>{offset:+d}</b> frames"
                + (f" — the {late} camera started {abs(offset) / fps:.2f} s later"
                   if offset else " (not synced yet?)"))
        else:
            self.sync_label.setText("Open both camera videos first.")

        have_both = self.images["left"] is not None and self.images["right"] is not None
        expect = self.expected_side()
        for s in ("left", "right"):
            active = have_both and s == expect
            self.views[s].setStyleSheet(
                f"QGraphicsView {{ border:4px solid {YELLOW.name() if active else '#444'}; }}")
            frame = f"  (frame {self.shown[s]})" if self.shown[s] is not None else ""
            self.view_titles[s].setText(
                f"<b>{s.upper()} camera</b>{frame}" + ("  —  click here next" if active else ""))
            self.view_titles[s].setStyleSheet(
                f"QLabel {{ font-size:14px; color:{'#B08800' if active else 'palette(text)'}; }}")

        if not have_both:
            missing = [s.upper() for s in ("left", "right") if self.images[s] is None]
            msg = (f"<b>First:</b> open the {' and '.join(missing)} camera file"
                   f"{'s' if len(missing) > 1 else ''} with the buttons at the top.")
        else:
            kind = self.mode.upper()
            color = KIND_COLOR[self.mode].name()
            number = len(self.pairs) + (0 if expect == "right" else 1)
            if expect == "left":
                msg = (f"<b>Click a <span style='color:{color}'>{kind}</span> point on the "
                       f"LEFT image</b> (point {number}).")
                if self.mode == "ground" and n_ground >= GOOD_GROUND and n_high == 0:
                    msg += "<br>Enough ground points. Press <b>H</b> to switch to high points."
                elif self.mode == "ground":
                    msg += "<br><small>Exact spots, spread out — not all on one line.</small>"
                else:
                    msg += "<br><small>Fence tops, poles, heads, treetops.</small>"
                if n_ground >= MIN_GROUND and n_high >= MIN_HIGH:
                    msg += "<br>You can press <b>Run check</b> now, or add more points."
            else:
                msg = (f"<b>Now click the SAME spot on the RIGHT image</b> (point {number}).<br>"
                       "<small>Zoom with the mouse wheel for accuracy.</small>")
        self.next_label.setText(msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
