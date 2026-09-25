"""Where the 3D stage reads footage and tracks from, and writes its results.

All three live outside the repository, because everything derived from the match
footage stays off version control. Defaults assume the footage sits next to the
repository; each can be overridden with an environment variable.

    FMM_FOOTAGE   the four source videos          (default ../raw_footage)
    FMM_TRACKS    tracker output, one folder per run (default ../outputs/tracking)
    FMM_WORK      calibration, sync and 3D results   (default ../outputs/mapping3d)
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)

FOOTAGE = os.environ.get("FMM_FOOTAGE", os.path.join(PARENT, "raw_footage"))
TRACKS = os.environ.get("FMM_TRACKS", os.path.join(PARENT, "outputs", "tracking"))
WORK = os.environ.get("FMM_WORK", os.path.join(PARENT, "outputs", "mapping3d"))
