"""Fine-tune YOLO26 on the reviewed footage.

Defaults are set for a small dataset cut from one pitch: a moderate model size, a
low learning rate, and augmentation that avoids shrinking an already tiny ball.

Example:
    python scripts/finetune_yolo26.py --data ../datasets/ours_tiled/data.yaml \
        --name tiled_m --model yolo26m.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--project", type=Path, default=Path("../training_runs"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze this many leading layers; helps when data is scarce",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        project=str(args.project.resolve()),
        name=args.name,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze or None,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        warmup_epochs=5,
        # The ball is ~34px; scaling down hard would erase it, so keep the range tight
        # and lean on colour and flip augmentation instead.
        scale=0.3,
        mosaic=1.0,
        close_mosaic=15,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        plots=True,
        seed=0,
        verbose=True,
    )

    metrics = model.val(data=str(args.data.resolve()), imgsz=args.imgsz, split="test")
    print("\n--- test split ---")
    print(f"mAP@50    {metrics.box.map50:.4f}")
    print(f"mAP@50-95 {metrics.box.map:.4f}")
    for i, name in metrics.names.items():
        if i < len(metrics.box.ap50):
            print(f"  {name:<8} AP@50 {metrics.box.ap50[i]:.4f}")
    print(f"\nweights: {args.project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
