from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Yolo_train" / "data.yaml").exists():
            return candidate
    return start


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return _find_project_root(Path(sys.executable).resolve().parent)
    return _find_project_root(Path(__file__).resolve().parent.parent)


PROJECT_ROOT = project_root()
YOLO_DIR = PROJECT_ROOT / "Yolo_train"
DEFAULT_DATA = YOLO_DIR / "data.yaml"
DEFAULT_PROJECT = YOLO_DIR / "runs"
RUNTIME_DATA = YOLO_DIR / "data.runtime.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detection model for Sobel edge images."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="YOLO dataset YAML path. Default: Yolo_train/data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLO model or checkpoint. Default: yolov8n.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs. Default: 50.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Default: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size. Default: 4.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers. Default: 0 for Windows reliability.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Training output folder. Default: Yolo_train/runs.",
    )
    parser.add_argument(
        "--name",
        default="sobel_yolo",
        help="Run name under the project folder. Default: sobel_yolo.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example cpu, 0, or 0,1. Default: auto.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the selected checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    data_path = _runtime_dataset_yaml(args.data)
    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "resume": args.resume,
    }
    if args.device:
        train_args["device"] = args.device

    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    print(f"Training complete: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    return 0


def _runtime_dataset_yaml(data_path: Path) -> Path:
    if data_path.resolve() != DEFAULT_DATA.resolve():
        return data_path

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["path"] = str(YOLO_DIR)
    RUNTIME_DATA.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return RUNTIME_DATA


if __name__ == "__main__":
    raise SystemExit(main())
