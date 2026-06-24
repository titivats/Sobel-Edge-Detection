from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from file_io import atomic_write_image
from realtime_config import IMAGE_EXTENSIONS
from realtime_overlay import BoxResult, draw_overlay, make_sobel_bgr, status_from_classes
from realtime_product_info import (
    CsvPointCounter,
    ProductCsvIndex,
    ProductInfo,
    ProductInfoCache,
    timestamp_from_name,
)


@dataclass(frozen=True)
class PredictionView:
    image_path: Path
    output_path: Path
    status: str
    point_number: int
    class_names: list[str]
    product_info: ProductInfo
    image_bgr: np.ndarray


class RealtimePredictor:
    def __init__(
        self,
        image_dir: Path,
        csv_dir: Path,
        model_path: Path,
        output_dir: Path,
        confidence: float,
        sobel_threshold_ratio: float,
    ) -> None:
        from ultralytics import YOLO

        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if sobel_threshold_ratio <= 0:
            raise ValueError("Sobel threshold ratio must be greater than 0")

        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.confidence = confidence
        self.sobel_threshold_ratio = sobel_threshold_ratio
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.processed: dict[Path, int] = {}
        self.point_counter = CsvPointCounter()
        self.csv_index = ProductCsvIndex(csv_dir)
        self.product_info_cache = ProductInfoCache()

    def update_paths(self, image_dir: Path, csv_dir: Path, output_dir: Path) -> None:
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.processed.clear()
        self.point_counter = CsvPointCounter()
        self.csv_index.update_dir(csv_dir)
        self.product_info_cache.clear()

    def scan(self) -> list[PredictionView]:
        return list(self.scan_iter())

    def scan_iter(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for image_path, modified in self.changed_images():
            view = self.predict_image(image_path)
            self.processed[image_path] = modified
            yield view

    def changed_images(self) -> list[tuple[Path, int]]:
        changed = []
        for image_path, modified in self.iter_image_candidates():
            if self.processed.get(image_path) == modified:
                continue
            changed.append((image_path, modified))
        return sorted(
            changed, key=lambda item: (timestamp_from_name(item[0]) or datetime.min, item[0].name)
        )

    def iter_image_candidates(self):
        if self.image_dir.is_file():
            if self.image_dir.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    yield self.image_dir, self.image_dir.stat().st_mtime_ns
                except OSError:
                    return
            return

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.image_dir}")

        stack = [self.image_dir]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                path = Path(entry.path)
                                if path.suffix.lower() in IMAGE_EXTENSIONS:
                                    yield path, entry.stat(follow_symlinks=False).st_mtime_ns
                        except OSError:
                            continue
            except OSError:
                continue

    def predict_image(self, image_path: Path) -> PredictionView:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        product_info = self.product_info_cache.read(self.csv_index.match(image_path))
        point_number = self.point_counter.next_point(product_info.csv_path)
        sobel_bgr = make_sobel_bgr(image_bgr, self.sobel_threshold_ratio)
        results = self.model.predict(sobel_bgr, conf=self.confidence, verbose=False)
        result = results[0]
        boxes: list[BoxResult] = []
        class_names = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            label = str(self.class_names[class_id])
            confidence = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, label, confidence))
            class_names.append(label)

        status = status_from_classes(class_names)
        annotated = draw_overlay(sobel_bgr, boxes, status, point_number, product_info)
        csv_stem = product_info.csv_path.stem if product_info.csv_path else "no_csv"
        output_path = (
            self.output_dir / f"{csv_stem}_point{point_number:03d}_{image_path.stem}_predicted.png"
        )
        atomic_write_image(output_path, annotated)
        return PredictionView(
            image_path=image_path,
            output_path=output_path,
            status=status,
            point_number=point_number,
            class_names=class_names,
            product_info=product_info,
            image_bgr=annotated,
        )
