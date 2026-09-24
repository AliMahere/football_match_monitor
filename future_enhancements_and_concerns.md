# Future enhancements and concerns

Written alongside the detector work. Everything here is either a limitation we know
about and have not solved, or a next step we would take with more time.

## Concerns about the current detector results

**The ball numbers rest on very few examples.** The held-out set contains 10 ball
instances on the left camera and 9 on the right. The fine-tuned model scores 0.703 and
1.000 AP on those. "1.000" means nine balls were found, not that ball detection is
solved. Any ball figure quoted from this work needs its denominator beside it.

**The test split is not a real generalisation test.** Frames were sampled every 20
frames from a single continuous passage, and the held-out frames are simply the last
eleven. At 45 fps that puts them under half a second from the training frames — same
pitch, same session, same light, same players. The result shows the model fits this
footage, not that it transfers to another match.

**There is no separate validation set.** With 53 reviewed frames per camera there was
not enough data for three splits, so validation and test are the same images. Early
stopping therefore selected the checkpoint using the same frames the final numbers are
reported on. The numbers are optimistic by an unknown margin.

**The fine-tuned model over-predicts players.** It reports 117 boxes where there are
102, and 119 where there are 101, holding precision at 0.81–0.84 against the
off-the-shelf model's tighter count. Detection metrics tolerate this, but tracking will
not: every spurious box is a chance to start a false identity that then has to be merged
away.

**The left camera is consistently worse than the right**, on every model and every
metric, from the first evaluation onwards. We have not established why. Until we do, any
figure averaged across the two cameras hides a real asymmetry.

**All training data comes from one session.** One pitch, one time of day, one set of
kit colours, two fixed viewpoints. Nothing in the data teaches the model about other
grounds, floodlighting, rain, or different bib colours.

## Concerns about the 3D requirement

**The two cameras are not a conventional stereo pair.** They sit side by side at
midfield and cover opposite halves of the pitch, overlapping only around the centre
circle. Triangulation only works inside that overlap, and if the two optical centres are
close together the depth estimate is poorly conditioned even there. The baseline needs
measuring before any triangulation result is trusted.

**The ball is usually visible to only one camera**, because each sees roughly half the
pitch. Stereo ball height is therefore unavailable for most of the match, whatever the
baseline turns out to be.

**The pitch is not a standard size.** It is a small-sided caged ground, so the usual
trick of fitting known full-size pitch dimensions does not apply. Dimensions have to be
recovered some other way, for example from the goals, which appear to be standard.

**The lenses distort noticeably.** Touchlines and the barrier wall curve visibly. A
plain homography assumes a pinhole camera and will fit worst at the frame edges, which
is exactly where the overlap region between the two cameras sits.

**The two clips are not the same length**, 13677 frames against 13556. The offset
between them has to be established before any frame is treated as simultaneous across
the pair.

## Performance concerns

**Tiled inference costs about 2.4 seconds per 4K frame**, against roughly 0.1 seconds
for a single downscaled pass. Tiling is what makes the ball detectable, so this is not
optional as things stand, but it is far from real time on five minutes of 45 fps
footage.

## Future enhancements

**Label more frames, chosen for the ball rather than by stride.** The current frames
were sampled at a fixed interval, so ball examples arrived by luck. Deliberately
selecting frames where the ball is airborne, blurred, near a line, or against a bright
background would target the failure modes directly.

**Use motion, not just appearance, for the ball.** Every appearance-only ball detector
we tested failed on this footage, and the recurring false positive was the penalty spot
— a white circle the size of a ball that never moves. A temporal detector such as WASB
takes several consecutive frames as input and would reject a stationary mark by
construction, rather than needing to be taught it.

**Restrict detection to the pitch.** Several false positives sit outside the playing
area entirely. Masking to the pitch polygon removes that class of error outright, and
the pitch geometry is needed for the 3D work anyway.

**Track the ball instead of re-finding it.** Searching a small window around the
previous position, and only falling back to a full tiled sweep when the track is lost,
would cut the per-frame cost dramatically while improving continuity.

**Bring the inference cost down.** Export to TensorRT, try a smaller backbone, and tile
adaptively rather than uniformly — most tiles contain only grass.

**Calibrate from pitch lines with lens distortion included.** PnLCalib recovers full
camera parameters from field lines and added distortion optimisation, which suits these
wide, curved views better than a plain homography.

**Get ball height monocularly, not only by triangulation.** Apparent ball diameter
against its known real size gives a depth estimate from a single view, which works
across the whole pitch rather than only the narrow overlap band. Triangulation in the
overlap can then serve as a check on it.

**Validate the 3D pipeline against data that has ground truth.** ISSIA-3D provides 3D
ball positions recovered from synchronised fixed cameras calibrated from field lines —
the same problem as ours. Running our method against it would turn our 3D claims into a
measured error rather than an assertion.

**Choose and evaluate a tracker.** Detection is settled enough to build on; the stable
identity requirement needs a tracker on top, and that choice has not been made or
measured yet.

**Test across cameras and sessions.** Training on one camera and testing on the other
would give a first honest read on generalisation, well before any second match is
available.
