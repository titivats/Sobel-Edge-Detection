from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2

from image_file_discovery import find_images
from sobel_edge_output import save_sobel_edge_image
from sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display


DEFAULT_INPUT_PATH = Path("SepData") / "Picture"
DEFAULT_OUTPUT_PATH = Path("Output_files")

MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


@dataclass(frozen=True)
class SobelOutput:
    image_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Sobel edge detection images from input images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input image file or directory. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory for Sobel edge images.",
    )
    parser.add_argument(
        "--sobel-threshold-ratio",
        type=float,
        default=0.12,
        help="Sobel threshold ratio used for edge masks. Default: 0.12.",
    )
    parser.add_argument(
        "--edge-close-kernel",
        type=int,
        default=1,
        help=(
            "Morphology close kernel size used to connect broken Sobel edge gaps. "
            "Use 1 to disable. Default: 1."
        ),
    )
    parser.add_argument(
        "--edge-close-iterations",
        type=int,
        default=0,
        help="Morphology close iterations for Sobel edge repair. Default: 0.",
    )
    parser.add_argument(
        "--edge-dilate-iterations",
        type=int,
        default=0,
        help="Dilation iterations after Sobel edge repair. Default: 0.",
    )
    parser.add_argument(
        "--display-edge-thickness",
        type=int,
        default=1,
        help=(
            "Thickness multiplier for saved Sobel display images only. "
            "Use 1 for raw Sobel display. Default: 1."
        ),
    )
    parser.add_argument(
        "--edge-view",
        choices=("x", "y", "all"),
        default="all",
        help="Sobel edge view to save: x, y, or all combined edges. Default: all.",
    )
    return parser.parse_args()


def daily_output_dir(base_output_dir: Path, run_date: date | None = None) -> Path:
    output_date = run_date or date.today()
    daily_folder_name = (
        f"{output_date.day:02d}-{MONTH_NAMES[output_date.month - 1]}-{output_date.year}"
    )
    if base_output_dir.name == daily_folder_name:
        return base_output_dir
    return base_output_dir / daily_folder_name


def create_sobel_edge_output(
    image_path: Path,
    output_dir: Path,
    sobel_threshold_ratio: float,
    edge_close_kernel: int,
    edge_close_iterations: int,
    edge_dilate_iterations: int,
    display_edge_thickness: int,
    edge_view: str,
) -> SobelOutput:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    sobel_x_edges, sobel_y_edges, sobel_all_edges = create_sobel_edge_masks(
        image,
        sobel_threshold_ratio,
        edge_close_kernel,
        edge_close_iterations,
        edge_dilate_iterations,
    )
    edge_by_view = {
        "x": sobel_x_edges,
        "y": sobel_y_edges,
        "all": sobel_all_edges,
    }
    display_edge = thicken_edge_for_display(
        edge_by_view[edge_view],
        display_edge_thickness,
    )
    output_path = save_sobel_edge_image(
        output_dir=output_dir,
        image_stem=image_path.stem,
        display_edge=display_edge,
    )
    return SobelOutput(image_path=image_path, output_path=output_path)


def main() -> int:
    args = parse_args()
    _validate_args(args)

    output_dir = daily_output_dir(args.output)
    images = find_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    outputs: list[SobelOutput] = []
    errors: list[tuple[Path, str]] = []
    for image_path in images:
        try:
            outputs.append(
                create_sobel_edge_output(
                    image_path=image_path,
                    output_dir=output_dir,
                    sobel_threshold_ratio=args.sobel_threshold_ratio,
                    edge_close_kernel=args.edge_close_kernel,
                    edge_close_iterations=args.edge_close_iterations,
                    edge_dilate_iterations=args.edge_dilate_iterations,
                    display_edge_thickness=args.display_edge_thickness,
                    edge_view=args.edge_view,
                )
            )
        except ValueError as exc:
            errors.append((image_path, str(exc)))

    if not outputs:
        raise RuntimeError("No Sobel edge images could be created. See input images.")

    print(f"Processed {len(outputs)} image(s)")
    if errors:
        print(f"Skipped {len(errors)} image(s)")
    print(f"Edge view: {args.edge_view}")
    print(
        "Edge repair: "
        f"close_kernel={args.edge_close_kernel}, "
        f"close_iterations={args.edge_close_iterations}, "
        f"dilate_iterations={args.edge_dilate_iterations}"
    )
    print(f"Sobel edge images: {(output_dir / 'sobel_edges').resolve()}")
    for output in outputs:
        print(f"{output.image_path.name}: {output.output_path.resolve()}")
    for image_path, error_message in errors:
        print(f"{image_path.name}: skipped, {error_message}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.sobel_threshold_ratio <= 0:
        raise ValueError("--sobel-threshold-ratio must be greater than 0")
    if args.edge_close_kernel < 1:
        raise ValueError("--edge-close-kernel must be greater than or equal to 1")
    if args.edge_close_iterations < 0:
        raise ValueError("--edge-close-iterations must be greater than or equal to 0")
    if args.edge_dilate_iterations < 0:
        raise ValueError("--edge-dilate-iterations must be greater than or equal to 0")
    if args.display_edge_thickness < 1:
        raise ValueError("--display-edge-thickness must be greater than or equal to 1")


if __name__ == "__main__":
    raise SystemExit(main())
