from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Measurement:
    image_path: Path
    width_px: int
    height_px: int
    contour_count: int
    edge_pixels: int
    foreground_pixels: int
    area_mm2: float
    total_length_px: float
    total_length_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Sobel edges and measure total edge contour length in millimeters."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"E:\Project_Edge_detection\SepData"),
        help="Input image file or directory. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory for CSV and generated images.",
    )
    parser.add_argument(
        "--mode",
        choices=("foreground", "edges"),
        default="foreground",
        help=(
            "foreground separates colored objects from a black background; "
            "edges measures all Sobel edges. Default: foreground."
        ),
    )
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument(
        "--mm-per-pixel",
        type=float,
        help="Calibration scale, for example 0.05 means each pixel is 0.05 mm.",
    )
    calibration.add_argument(
        "--pixels-per-mm",
        type=float,
        help="Calibration scale, for example 20 means 20 pixels equals 1 mm.",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.25,
        help="Edge threshold as a fraction of the strongest Sobel magnitude. Default: 0.25.",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=5,
        help="Odd Gaussian blur kernel size before Sobel. Use 0 to disable. Default: 5.",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=100.0,
        help="Ignore contours smaller than this area in pixels. Default: 100.",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=35,
        help=(
            "Maximum grayscale value treated as black background in foreground mode. Default: 35."
        ),
    )
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=20,
        help=(
            "Minimum HSV saturation treated as colored foreground in foreground mode. Default: 20."
        ),
    )
    return parser.parse_args()


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_blur_kernel(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def sobel_edge_mask(
    image: np.ndarray,
    threshold_ratio: float,
    blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_kernel:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    max_magnitude = float(magnitude.max())
    if max_magnitude <= 0:
        return np.zeros(gray.shape, dtype=np.uint8), gray

    threshold = max_magnitude * threshold_ratio
    edge_mask = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)

    # Close small gaps and remove isolated single-pixel noise before contour measurement.
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)

    return edge_mask, gray


def foreground_mask_from_black_background(
    image: np.ndarray,
    black_threshold: int,
    min_saturation: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    value_mask = gray > black_threshold
    saturation_mask = hsv[:, :, 1] >= min_saturation
    foreground = np.where(value_mask & saturation_mask, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    return foreground


def boundary_mask_from_foreground_and_sobel(
    foreground_mask: np.ndarray,
    sobel_mask: np.ndarray,
) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(foreground_mask, cv2.MORPH_GRADIENT, kernel)
    nearby_sobel = cv2.dilate(sobel_mask, kernel, iterations=1)
    return cv2.bitwise_and(boundary, nearby_sobel)


def largest_foreground_contours(
    foreground_mask: np.ndarray,
    min_contour_area: float,
) -> list[np.ndarray]:
    contours, _hierarchy = cv2.findContours(
        foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]


def measure_image(
    image_path: Path,
    output_dir: Path,
    mm_per_pixel: float,
    mode: str,
    threshold_ratio: float,
    blur_kernel: int,
    min_contour_area: float,
    black_threshold: int,
    min_saturation: int,
) -> Measurement:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    sobel_mask, _gray = sobel_edge_mask(image, threshold_ratio, blur_kernel)

    if mode == "foreground":
        foreground_mask = foreground_mask_from_black_background(
            image, black_threshold, min_saturation
        )
        kept_contours = largest_foreground_contours(foreground_mask, min_contour_area)
        edge_mask = boundary_mask_from_foreground_and_sobel(foreground_mask, sobel_mask)
        total_length_px = float(
            sum(cv2.arcLength(contour, closed=True) for contour in kept_contours)
        )
        foreground_pixels = int(np.count_nonzero(foreground_mask))
    else:
        foreground_mask = np.zeros(sobel_mask.shape, dtype=np.uint8)
        contours, _hierarchy = cv2.findContours(
            sobel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        kept_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= min_contour_area
        ]
        edge_mask = sobel_mask
        total_length_px = float(
            sum(cv2.arcLength(contour, closed=False) for contour in kept_contours)
        )
        foreground_pixels = 0

    total_length_mm = total_length_px * mm_per_pixel
    area_mm2 = foreground_pixels * (mm_per_pixel**2)

    relative_name = image_path.stem
    edge_output = output_dir / "edges" / f"{relative_name}_edges.png"
    mask_output = output_dir / "masks" / f"{relative_name}_mask.png"
    annotated_output = output_dir / "annotated" / f"{relative_name}_annotated.png"
    edge_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    annotated_output.parent.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    overlay = annotated.copy()
    overlay[foreground_mask > 0] = (0, 180, 0)
    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
    cv2.drawContours(annotated, kept_contours, -1, (0, 0, 255), 2)
    cv2.putText(
        annotated,
        f"Boundary: {total_length_mm:.3f} mm  Area: {area_mm2:.3f} mm2",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(edge_output), edge_mask)
    cv2.imwrite(str(mask_output), foreground_mask)
    cv2.imwrite(str(annotated_output), annotated)

    height_px, width_px = edge_mask.shape
    return Measurement(
        image_path=image_path,
        width_px=width_px,
        height_px=height_px,
        contour_count=len(kept_contours),
        edge_pixels=int(np.count_nonzero(edge_mask)),
        foreground_pixels=foreground_pixels,
        area_mm2=area_mm2,
        total_length_px=total_length_px,
        total_length_mm=total_length_mm,
    )


def write_csv(measurements: list[Measurement], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "width_px",
                "height_px",
                "contour_count",
                "edge_pixels",
                "foreground_pixels",
                "area_mm2",
                "total_length_px",
                "total_length_mm",
            ],
        )
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(
                {
                    "image": str(measurement.image_path),
                    "width_px": measurement.width_px,
                    "height_px": measurement.height_px,
                    "contour_count": measurement.contour_count,
                    "edge_pixels": measurement.edge_pixels,
                    "foreground_pixels": measurement.foreground_pixels,
                    "area_mm2": f"{measurement.area_mm2:.3f}",
                    "total_length_px": f"{measurement.total_length_px:.3f}",
                    "total_length_mm": f"{measurement.total_length_mm:.3f}",
                }
            )


def main() -> int:
    args = parse_args()
    if args.threshold_ratio <= 0 or args.threshold_ratio > 1:
        raise ValueError("--threshold-ratio must be greater than 0 and less than or equal to 1")

    if args.mm_per_pixel is not None:
        if args.mm_per_pixel <= 0:
            raise ValueError("--mm-per-pixel must be greater than 0")
        mm_per_pixel = args.mm_per_pixel
    else:
        if args.pixels_per_mm <= 0:
            raise ValueError("--pixels-per-mm must be greater than 0")
        mm_per_pixel = 1.0 / args.pixels_per_mm

    blur_kernel = normalize_blur_kernel(args.blur)
    images = find_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    measurements = [
        measure_image(
            image_path=image_path,
            output_dir=args.output,
            mm_per_pixel=mm_per_pixel,
            mode=args.mode,
            threshold_ratio=args.threshold_ratio,
            blur_kernel=blur_kernel,
            min_contour_area=args.min_contour_area,
            black_threshold=args.black_threshold,
            min_saturation=args.min_saturation,
        )
        for image_path in images
    ]

    csv_path = args.output / "measurements.csv"
    write_csv(measurements, csv_path)

    print(f"Processed {len(measurements)} image(s)")
    print(f"Calibration: {mm_per_pixel:.8f} mm/pixel")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Masks: {(args.output / 'masks').resolve()}")
    print(f"Edge images: {(args.output / 'edges').resolve()}")
    print(f"Annotated images: {(args.output / 'annotated').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
