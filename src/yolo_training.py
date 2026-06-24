from __future__ import annotations

import argparse
import configparser
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from file_io import atomic_write_text

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class TrainingSettings:
    images: Path
    labels: Path
    classes: Path
    model: str
    output: Path
    run_name: str
    epochs: int
    image_size: int
    batch: int
    workers: int
    device: str
    patience: int
    validation_split: float
    seed: int
    cache: bool
    resume: bool
    exist_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model using values from yolo/setting.txt."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "yolo" / "setting.txt",
        help="Path to the INI-style setting file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate settings and dataset without starting training.",
    )
    return parser.parse_args()


def load_settings(settings_path: Path) -> TrainingSettings:
    settings_path = settings_path.resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    parser = configparser.ConfigParser()
    parser.read(settings_path, encoding="utf-8")
    if "training" not in parser:
        raise ValueError(f"Missing [training] section in: {settings_path}")

    section = parser["training"]
    base_dir = settings_path.parent
    model_value = _required(section, "model")
    model = _resolve_model(base_dir, model_value)

    settings = TrainingSettings(
        images=_resolve_path(base_dir, _required(section, "images")),
        labels=_resolve_path(base_dir, _required(section, "labels")),
        classes=_resolve_path(base_dir, _required(section, "classes")),
        model=model,
        output=_resolve_path(base_dir, section.get("output", "runs")),
        run_name=section.get("run_name", "sobel_fine_tune").strip(),
        epochs=section.getint("epochs", 50),
        image_size=section.getint("image_size", 640),
        batch=section.getint("batch", 8),
        workers=section.getint("workers", 0),
        device=section.get("device", "cpu").strip(),
        patience=section.getint("patience", 20),
        validation_split=section.getfloat("validation_split", 0.20),
        seed=section.getint("seed", 42),
        cache=section.getboolean("cache", False),
        resume=section.getboolean("resume", False),
        exist_ok=section.getboolean("exist_ok", True),
    )
    _validate_settings(settings)
    return settings


def validate_dataset(
    settings: TrainingSettings,
) -> tuple[list[Path], list[str], dict[int, int], dict[Path, tuple[int, ...]]]:
    if not settings.images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {settings.images}")
    if not settings.labels.is_dir():
        raise FileNotFoundError(f"Label directory not found: {settings.labels}")
    if not settings.classes.is_file():
        raise FileNotFoundError(f"Class file not found: {settings.classes}")

    class_names = [
        line.strip()
        for line in settings.classes.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"No classes found in: {settings.classes}")

    images = sorted(
        path
        for path in settings.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in: {settings.images}")

    missing_labels = [
        image.name for image in images if not (settings.labels / f"{image.stem}.txt").is_file()
    ]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"Missing labels for {len(missing_labels)} image(s): {preview}")

    image_stems = {image.stem for image in images}
    orphan_labels = [
        label.name for label in settings.labels.glob("*.txt") if label.stem not in image_stems
    ]
    if orphan_labels:
        preview = ", ".join(orphan_labels[:5])
        raise ValueError(f"Labels without matching images: {len(orphan_labels)} file(s): {preview}")

    class_counts, class_signatures = _validate_class_ids(
        settings.labels,
        len(class_names),
        images,
    )
    return images, class_names, class_counts, class_signatures


def write_dataset_yaml(
    settings_path: Path,
    settings: TrainingSettings,
    class_names: list[str],
    images: list[Path],
    class_signatures: dict[Path, tuple[int, ...]],
) -> Path:
    yolo_dir = settings_path.resolve().parent
    generated_dir = settings.output / ".config"
    generated_dir.mkdir(parents=True, exist_ok=True)
    train_images, validation_images = _split_images(
        images,
        class_signatures,
        settings.validation_split,
        settings.seed,
    )
    train_list = generated_dir / "train.txt"
    validation_list = generated_dir / "validation.txt"
    _write_image_list(train_list, train_images)
    _write_image_list(validation_list, validation_images)

    data_path = generated_dir / "data.yaml"
    dataset = {
        "path": str(yolo_dir),
        "train": str(train_list),
        "val": str(validation_list),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    atomic_write_text(
        data_path,
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True),
    )
    return data_path


def train(settings: TrainingSettings, data_path: Path) -> Path:
    from ultralytics import YOLO

    settings.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(settings.model)
    try:
        results = model.train(
            data=str(data_path),
            epochs=settings.epochs,
            imgsz=settings.image_size,
            batch=settings.batch,
            workers=settings.workers,
            device=settings.device,
            patience=settings.patience,
            cache=settings.cache,
            resume=settings.resume,
            project=str(settings.output),
            name=settings.run_name,
            exist_ok=settings.exist_ok,
        )
    finally:
        settings.labels.with_suffix(".cache").unlink(missing_ok=True)
    return Path(results.save_dir)


def main() -> int:
    args = parse_args()
    settings_path = args.settings.resolve()
    settings = load_settings(settings_path)
    images, class_names, class_counts, class_signatures = validate_dataset(settings)
    data_path = write_dataset_yaml(
        settings_path,
        settings,
        class_names,
        images,
        class_signatures,
    )

    print(f"Settings: {settings_path}")
    print(f"Images: {len(images)}")
    print(f"Classes: {', '.join(class_names)}")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id}={class_name}: {class_counts.get(class_id, 0)} objects")
        if class_counts.get(class_id, 0) == 0:
            print(f"  WARNING: Class '{class_name}' has no labeled objects.")
    print(f"Model: {settings.model}")
    print(f"Device: {settings.device}")
    print(f"Dataset YAML: {data_path}")
    print(f"Output: {settings.output / settings.run_name}")

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return 0

    save_dir = train(settings, data_path)
    print(f"Training complete: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    return 0


def _required(section: configparser.SectionProxy, key: str) -> str:
    value = section.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing required setting: {key}")
    return value


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_model(base_dir: Path, value: str) -> str:
    looks_like_path = (
        Path(value).is_absolute() or value.startswith((".", "~")) or "/" in value or "\\" in value
    )
    if not looks_like_path:
        return value
    model_path = _resolve_path(base_dir, value)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return str(model_path)


def _validate_settings(settings: TrainingSettings) -> None:
    if not settings.run_name:
        raise ValueError("run_name must not be empty")
    if settings.epochs < 1:
        raise ValueError("epochs must be greater than 0")
    if settings.image_size < 32:
        raise ValueError("image_size must be at least 32")
    if settings.batch == 0 or settings.batch < -1:
        raise ValueError("batch must be -1 or greater than 0")
    if settings.workers < 0:
        raise ValueError("workers must be 0 or greater")
    if settings.patience < 0:
        raise ValueError("patience must be 0 or greater")
    if not 0.0 < settings.validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")


def _validate_class_ids(
    labels_dir: Path,
    class_count: int,
    images: list[Path],
) -> tuple[dict[int, int], dict[Path, tuple[int, ...]]]:
    class_counts = {class_id: 0 for class_id in range(class_count)}
    class_signatures: dict[Path, tuple[int, ...]] = {}
    images_by_stem = {image.stem: image for image in images}
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_by_stem[label_path.stem]
        image_class_ids: list[int] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}; " "expected 5 values"
                )
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value at {label_path}:{line_number}") from exc
            if not 0 <= class_id < class_count:
                raise ValueError(
                    f"Class ID {class_id} at {label_path}:{line_number} "
                    f"is outside 0..{class_count - 1}"
                )
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(
                    f"Coordinates at {label_path}:{line_number} must be between 0 and 1"
                )
            class_counts[class_id] += 1
            image_class_ids.append(class_id)
        class_signatures[image_path] = tuple(sorted(set(image_class_ids)))
    return class_counts, class_signatures


def _split_images(
    images: list[Path],
    class_signatures: dict[Path, tuple[int, ...]],
    validation_split: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    randomizer = random.Random(seed)
    groups: dict[tuple[int, ...], list[Path]] = {}
    for image in images:
        groups.setdefault(class_signatures[image], []).append(image)

    train_images: list[Path] = []
    validation_images: list[Path] = []
    for class_signature in sorted(groups):
        group = sorted(groups[class_signature])
        randomizer.shuffle(group)
        validation_count = max(1, round(len(group) * validation_split))
        if len(group) > 1:
            validation_count = min(validation_count, len(group) - 1)
        else:
            validation_count = 0
        validation_images.extend(group[:validation_count])
        train_images.extend(group[validation_count:])

    if not train_images or not validation_images:
        raise ValueError("Dataset is too small to create train and validation splits")
    return sorted(train_images), sorted(validation_images)


def _write_image_list(output_path: Path, images: list[Path]) -> None:
    atomic_write_text(
        output_path,
        "\n".join(str(image.resolve()) for image in images) + "\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
