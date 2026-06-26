from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    export_folder = folder / "Export JSON from label-studio"
    project_root = folder.parent
    parser = argparse.ArgumentParser(
        description="Convert a Label Studio Choices export into a YOLO classification dataset."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Label Studio JSON export. Defaults to the newest JSON file in "
            "'Export JSON from label-studio'."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=project_root / "Output_Sobel",
        help="Folder containing the original images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=folder / "dataset",
        help="Output dataset folder.",
    )
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_folder = Path(__file__).resolve().parent / "Export JSON from label-studio"
    export_path = args.json or newest_export(export_folder)
    image_index = index_images(args.images.resolve())
    records = read_classifications(export_path.resolve())

    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split must be between 0 and 1")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)

    randomizer = random.Random(args.seed)
    class_counts: Counter[str] = Counter()
    missing: list[str] = []

    by_class: dict[str, list[tuple[str, Path]]] = {}
    for uploaded_name, class_name in records:
        source = find_source_image(uploaded_name, image_index)
        if source is None:
            missing.append(uploaded_name)
            continue
        by_class.setdefault(class_name, []).append((uploaded_name, source))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not match {len(missing)} exported image(s) under {args.images}: {preview}"
        )

    for class_name, items in sorted(by_class.items()):
        randomizer.shuffle(items)
        validation_count = max(1, round(len(items) * args.validation_split))
        if len(items) > 1:
            validation_count = min(validation_count, len(items) - 1)
        else:
            validation_count = 0

        for index, (_uploaded_name, source) in enumerate(items):
            split = "val" if index < validation_count else "train"
            destination = output / split / safe_class_name(class_name) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            class_counts[f"{split}/{class_name}"] += 1

    print(f"Export: {export_path}")
    print(f"Images: {args.images.resolve()}")
    print(f"Dataset: {output}")
    for key, count in sorted(class_counts.items()):
        print(f"{key}: {count}")
    print("Dataset preparation complete.")
    return 0


def newest_export(folder: Path) -> Path:
    exports = sorted(
        folder.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not exports:
        raise FileNotFoundError(f"No Label Studio JSON export found in: {folder}")
    return exports[0]


def read_classifications(export_path: Path) -> list[tuple[str, str]]:
    tasks = json.loads(export_path.read_text(encoding="utf-8-sig"))
    records: list[tuple[str, str]] = []

    for task in tasks:
        annotations = [
            annotation
            for annotation in task.get("annotations", [])
            if not annotation.get("was_cancelled", False)
        ]
        if not annotations:
            continue

        choices: list[str] = []
        for result in annotations[-1].get("result", []):
            choices.extend(result.get("value", {}).get("choices", []))

        if len(choices) != 1:
            raise ValueError(
                f"Task {task.get('id')} must contain exactly one classification choice"
            )

        uploaded_name = task.get("file_upload") or Path(
            task.get("data", {}).get("image", "")
        ).name
        if not uploaded_name:
            raise ValueError(f"Task {task.get('id')} has no image filename")
        records.append((uploaded_name, choices[0]))

    if not records:
        raise ValueError(f"No completed image classifications found in: {export_path}")
    return records


def index_images(folder: Path) -> dict[str, list[Path]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    index: dict[str, list[Path]] = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def find_source_image(uploaded_name: str, image_index: dict[str, list[Path]]) -> Path | None:
    candidates = [uploaded_name, UPLOAD_PREFIX.sub("", uploaded_name)]
    for candidate in candidates:
        matches = image_index.get(candidate.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple source images match: {candidate}")
    return None


def safe_class_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", value.strip())
    if not name:
        raise ValueError("Class name must not be empty")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
