#!/usr/bin/env bash
# Rebuild the reviewed dataset end to end, from the CVAT exports to a YOLO
# dataset under datasets/ours. Run from the repository root.
#
#   bash "bashes/generate dataset.sh"
#
# Inputs are the CVAT COCO exports of the pre-annotated frames, corrected by
# hand. Only frames from 1940 on were reviewed, so earlier ones are discarded.

set -euo pipefail

OUT_ROOT="../inference_out"
DATASET="../datasets/ours"
FIRST_REVIEWED_FRAME=1940

# 1. Keep the reviewed frames, collapse roles onto ball/player, split train/test.
for CAM in left right; do
    python3 scripts/prepare_labels.py \
        --cvat "${OUT_ROOT}/cvat_${CAM}_3zvbc/cvat ${CAM}.json" \
        --out "${OUT_ROOT}/labels_${CAM}" \
        --from-frame "${FIRST_REVIEWED_FRAME}"
done

# 2. Merge both cameras into a single YOLO dataset.
python3 scripts/build_dataset.py \
    --camera left "${OUT_ROOT}/labels_left" "${OUT_ROOT}/cvat_left_3zvbc/frames" \
    --camera right "${OUT_ROOT}/labels_right" "${OUT_ROOT}/cvat_right_3zvbc/frames" \
    --out "${DATASET}"
