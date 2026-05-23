from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .image_file_discovery import find_images
from .intrusion_measurement import measure_intrusion
from .measurement_csv_writer import write_csv
from .focus_zone import focus_zone_from_ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure material intrusion into black router background using Sobel "
            "edge detection on both X and Y axes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"E:\Project_Edge_detection\SepData\camera"),
        help="Input image file or directory. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_intrusion_sobel"),
        help="Output directory for CSV and Sobel measurement images.",
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
        "--black-threshold",
        type=int,
        default=55,
        help="Grayscale value equal or below this is treated as black background. Default: 55.",
    )
    parser.add_argument(
        "--min-span",
        type=int,
        default=30,
        help="Ignore row/column measurements where Sobel edges are closer than this. Default: 30.",
    )
    parser.add_argument(
        "--baseline-percentile",
        type=float,
        default=10.0,
        help="Percentile used to estimate normal black-background boundaries. Default: 10.",
    )
    parser.add_argument(
        "--sobel-threshold-ratio",
        type=float,
        default=0.12,
        help="Sobel threshold ratio used for edge masks. Default: 0.12.",
    )
    parser.add_argument(
        "--display-edge-thickness",
        type=int,
        default=2,
        help=(
            "Thickness multiplier for saved Sobel display images only. "
            "Does not change measurement values. Default: 2."
        ),
    )
    parser.add_argument(
        "--roi-margin",
        type=int,
        default=60,
        help="Extra pixels around the detected black area when searching Sobel edges. Default: 60.",
    )
    parser.add_argument(
        "--orientation",
        choices=("auto", "vertical", "horizontal"),
        default="auto",
        help=(
            "Black-background direction. vertical measures left/right; horizontal "
            "measures top/bottom. Default: auto."
        ),
    )
    parser.add_argument(
        "--black-edge-search-radius",
        type=int,
        default=18,
        help=(
            "Pixels around the original black-area boundary used to select the Sobel edge. "
            "Default: 18."
        ),
    )
    parser.add_argument(
        "--focus-x-min-ratio",
        type=float,
        default=0.20,
        help="Ignore image area left of this width ratio. Default: 0.20.",
    )
    parser.add_argument(
        "--focus-x-max-ratio",
        type=float,
        default=0.85,
        help="Ignore image area right of this width ratio. Default: 0.85.",
    )
    parser.add_argument(
        "--focus-y-min-ratio",
        type=float,
        default=0.15,
        help="Ignore image area above this height ratio. Default: 0.15.",
    )
    parser.add_argument(
        "--focus-y-max-ratio",
        type=float,
        default=0.85,
        help="Ignore image area below this height ratio. Default: 0.85.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mm_per_pixel is not None:
        if args.mm_per_pixel <= 0:
            raise ValueError("--mm-per-pixel must be greater than 0")
        mm_per_pixel = args.mm_per_pixel
    else:
        if args.pixels_per_mm <= 0:
            raise ValueError("--pixels-per-mm must be greater than 0")
        mm_per_pixel = 1.0 / args.pixels_per_mm

    if not 0 <= args.black_threshold <= 255:
        raise ValueError("--black-threshold must be between 0 and 255")
    if not 0 < args.baseline_percentile < 50:
        raise ValueError("--baseline-percentile must be greater than 0 and less than 50")
    if not 0 <= args.focus_x_min_ratio < args.focus_x_max_ratio <= 1:
        raise ValueError("--focus-x-min-ratio must be less than --focus-x-max-ratio within 0..1")
    if not 0 <= args.focus_y_min_ratio < args.focus_y_max_ratio <= 1:
        raise ValueError("--focus-y-min-ratio must be less than --focus-y-max-ratio within 0..1")

    images = find_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    first_image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Could not read image: {images[0]}")
    focus_zone = focus_zone_from_ratios(
        first_image.shape[:2],
        args.focus_x_min_ratio,
        args.focus_x_max_ratio,
        args.focus_y_min_ratio,
        args.focus_y_max_ratio,
    )

    measurements = [
        measure_intrusion(
            image_path=image_path,
            output_dir=args.output,
            mm_per_pixel=mm_per_pixel,
            black_threshold=args.black_threshold,
            min_span=args.min_span,
            baseline_percentile=args.baseline_percentile,
            sobel_threshold_ratio=args.sobel_threshold_ratio,
            roi_margin=args.roi_margin,
            orientation=args.orientation,
            display_edge_thickness=args.display_edge_thickness,
            black_edge_search_radius=args.black_edge_search_radius,
            focus_zone=focus_zone,
        )
        for image_path in images
    ]
    csv_path = args.output / "intrusion_measurements.csv"
    write_csv(measurements, csv_path)

    print(f"Processed {len(measurements)} image(s)")
    print(f"Calibration: {mm_per_pixel:.8f} mm/pixel")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Sobel edge images: {(args.output / 'sobel_edges').resolve()}")
    print(f"Annotated images: {(args.output / 'annotated').resolve()}")
    for m in measurements:
        print(
            f"{m.image_path.name}: "
            f"orientation={m.orientation}, "
            f"L={m.min_left_intrusion_px:.3f}/{m.max_left_intrusion_px:.3f}px, "
            f"R={m.min_right_intrusion_px:.3f}/{m.max_right_intrusion_px:.3f}px, "
            f"T={m.min_top_intrusion_px:.3f}/{m.max_top_intrusion_px:.3f}px, "
            f"B={m.min_bottom_intrusion_px:.3f}/{m.max_bottom_intrusion_px:.3f}px, "
            f"overall={m.max_intrusion_px:.3f}px "
            f"({m.max_intrusion_mm:.3f}mm, {m.max_intrusion_side})"
        )
    return 0
