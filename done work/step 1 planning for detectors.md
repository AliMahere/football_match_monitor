# Step 1 — Planning for detectors

## Understanding the requirements

The task asks for four things: tracking players with stable identities, tracking the
ball, estimating real-world positions in 3D from the stereo pair, and a short writeup.
Everything downstream depends on detection, so detection was settled first.

Looking at the footage changed some early assumptions. The two "stereo" cameras do not
film the same scene from two angles — they sit side by side at midfield and cover
opposite halves of the pitch, overlapping only around the centre circle. Both are 4K at
45 fps. The pitch is a small-sided caged artificial-turf ground, not a full-size pitch,
and the wide lenses bend the touchlines noticeably. Players appear large, around 400
pixels tall. The ball is about 50 pixels across.

## The goal

Find the best available detector for players and for the ball on this specific footage,
and only build our own if nothing off the shelf is good enough.

## Exploring what already exists

We searched Roboflow Universe, Hugging Face, GitHub and the SoccerNet literature. The
serious candidates were:

- YOLO26, generic COCO weights, no football training at all
- RF-DETR fine-tuned on SoccerNet-Tracking
- Roboflow's football-players-detection model (RF-DETR medium)
- Roboflow's dedicated ball-only detector
- Two YOLO11 models fine-tuned on football, one for players and one for the ball
- WASB, a temporal ball tracker, and FootAndBall, both trained on fixed-camera footage

Their published scores looked convincing — 99% player accuracy for one, 89% for a ball
detector — but they were all measured on broadcast television footage, which is nothing
like an elevated camera over a caged pitch. So we treated the published numbers as a
shortlist only, and measured everything ourselves.

## Testing the candidates on our own footage

We ran six configurations on the same six frames and looked at the output.

The result was the opposite of what the published scores predicted. Generic YOLO26 found
every player with tight, confident boxes. The football-specific YOLO11 player model found
almost nothing. Both dedicated ball detectors — including the one reporting 89% — found
the ball zero times, even when we ran them at full resolution on image tiles so the ball
was never shrunk. Three purpose-built ball detectors, three complete failures.

The only things that found the ball were generic COCO and the multi-class football
models, and the latter paid for it by repeatedly detecting the penalty spot as a ball. On
the left camera that single false detection appeared in 88 of 100 frames, always at the
same pixel, because the camera never moves.

The conclusion was that no off-the-shelf ball detector solves this, so the ball needs
either training on our own footage or a method that uses motion across frames.

## Building labelled data

With no usable ground truth for this pitch, we made our own. Roboflow's player model
pre-annotated 100 frames from each camera, sampled every 20 frames from the same starting
point in both, so the two cameras stay synchronised for the 3D work later.

Those pre-annotations were then corrected by hand in CVAT — locally, since the footage
cannot be uploaded anywhere. 53 frames per camera were fully reviewed: false ball
detections removed, missed players added, boxes tightened, and the unreliable goalkeeper
and referee labels folded into a single player class.

This gave 106 reviewed frames holding 1007 player boxes and 87 ball boxes, split into 84
training and 22 test images. The split is taken from the end of the clip rather than at
random, because frames less than half a second apart are nearly identical and a random
split would leak between training and test.

## Choosing the detector

We scored YOLO26 and the Roboflow player model against the corrected labels, grouping
everything into two classes because YOLO26 can only report people and sports balls.

The two models fail differently. The Roboflow model finds slightly more players and draws
tighter boxes, but invents about 25% more player boxes than really exist, dropping its
precision to around 0.80. YOLO26 predicts almost exactly the right number of players and
holds precision at 0.95 to 0.97, and it also scores better on the ball.

For tracking, precision matters more than a tight box: every invented detection is a
chance to start a false identity that then has to be untangled. Combined with YOLO26
being faster and better on the ball, it is the better base.

**Decision: fine-tune YOLO26 on the reviewed footage**, and treat the ball as the open
problem, since 87 labelled ball instances may not be enough to beat what the generic
weights already do.
