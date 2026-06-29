import argparse
import multiprocessing
from pathlib import Path

import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PASS/NG image classifier.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="pass_ng_classifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_dir = DATASET / "train"
    validation_dir = DATASET / "val"
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise RuntimeError(
            "Training dataset not found. Run python image_classification\\prepare_dataset.py "
            "after exporting Label Studio JSON."
        )
    train_classes = {path.name for path in train_dir.iterdir() if path.is_dir()}
    validation_classes = {path.name for path in validation_dir.iterdir() if path.is_dir()}
    classes = train_classes & validation_classes

    if len(classes) < 2:
        raise RuntimeError(
            "Training requires at least two classes in both dataset/train and dataset/val. "
            f"Found: {', '.join(sorted(classes)) or 'none'}. "
            "Label both PASS and NG images in Label Studio, export JSON, and run "
            "python image_classification\\prepare_dataset.py before training again."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but CUDA-enabled PyTorch is not available. "
            "Run scripts\\install_gpu_torch.ps1, then try python image_classification\\train.py again."
        )

    print(f"Training device: CUDA GPU 0 ({torch.cuda.get_device_name(0)})")
    model = YOLO(HERE / "yolo11n-cls.pt")
    model.train(
        data=DATASET,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,
        amp=False,
        project=HERE / "runs",
        name=args.name,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
