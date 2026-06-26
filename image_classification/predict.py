from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "pass_ng_classifier" / "weights" / "best.pt"
DEFAULT_SOURCE = HERE / "predict_images"
DEFAULT_OUTPUT = HERE / "prediction_results"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify images as PASS or NG.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    source = args.source.resolve()
    output = args.output.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not source.exists():
        source.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"Prediction folder was created. Put new images here and run again: {source}"
        )

    images = find_images(source)
    if not images:
        raise ValueError(f"No supported images found in: {source}")

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Prediction device: {device_name}")

    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "predictions.csv"
    rows: list[dict[str, str]] = []

    model = YOLO(model_path)
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        device=device,
        verbose=False,
        stream=True,
    )

    for source_path, result in zip(images, results, strict=True):
        if result.probs is None:
            raise RuntimeError("The selected model is not an image-classification model.")

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        predicted_class = result.names[class_id]
        destination = image_output / source_path.with_suffix(".jpg").name

        save_annotated_image(
            source_path,
            destination,
            predicted_class,
            confidence,
        )
        rows.append(
            {
                "filename": source_path.name,
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.6f}",
                "confidence_percent": f"{confidence * 100:.2f}",
            }
        )
        print(f"{source_path.name}: {predicted_class} ({confidence:.2%})")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "filename",
                "predicted_class",
                "confidence",
                "confidence_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV report: {csv_path}")
    print(f"Annotated images: {image_output}")
    return 0


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_annotated_image(
    source: Path,
    destination: Path,
    predicted_class: str,
    confidence: float,
) -> None:
    encoded = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode image: {source}")

    color = (0, 180, 0) if predicted_class.upper() == "PASS" else (0, 0, 255)
    text = f"{predicted_class} {confidence:.1%}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
        cv2.LINE_AA,
    )
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise OSError(f"Could not encode prediction image: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(destination)


if __name__ == "__main__":
    raise SystemExit(main())
