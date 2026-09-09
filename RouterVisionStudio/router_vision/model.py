"""Sobel-edge DINOv2 classifier for cut inspection images.

Approach: freeze a pretrained DINOv2 ViT and train only a linear head on top of
its embeddings.  With a few dozen labelled images that is the right shape of
model - fine-tuning the whole backbone would simply memorise the set.  Feature
Every image is converted to a deterministic Sobel edge-magnitude map before it
reaches DINOv2.  The preprocessing version and parameters are saved with the
head so a colour-image model can never be used accidentally on edge images.

The backbone is loaded through torch.hub and cached under the user's torch hub
directory; the first load needs internet, after that it works offline.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .guard import check_write_target

BACKBONES = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}
DEFAULT_BACKBONE = "dinov2_vits14"
DINOV2_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
INPUT_SIZE = 224  # must be a multiple of 14 for DINOv2
PREPROCESS_VERSION = "sobel-magnitude-v1"
MODEL_CLASSES = ("GOOD", "NG")
MIN_MODEL_VALIDATION_ACCURACY = 0.80
MIN_MODEL_CLASS_RECALL = 0.50
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class CropBox:
    """Region of the frame handed to the model, in source pixels.

    Defaults frame the routed channel in the middle of a 1440x1080 frame rather
    than squashing the whole image, so the model sees the cut at useful detail.
    """

    x0: int = 400
    y0: int = 290
    x1: int = 1040
    y1: int = 930

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0, x1 = max(0, min(self.x0, w - 1)), max(1, min(self.x1, w))
        y0, y1 = max(0, min(self.y0, h - 1)), max(1, min(self.y1, h))
        if x1 <= x0 or y1 <= y0:
            return image
        return image[y0:y1, x0:x1]


@dataclass
class SobelConfig:
    """Versioned edge preprocessing shared by training and production."""

    blur_ksize: int = 3
    blur_sigma: float = 0.0
    sobel_ksize: int = 3
    gradient_x_weight: float = 1.0
    gradient_y_weight: float = 1.0
    clip_percentile: float = 99.5
    edge_gain: float = 1.0
    noise_floor: int = 0

    def validated(self) -> "SobelConfig":
        blur = max(1, int(self.blur_ksize))
        sobel = max(1, int(self.sobel_ksize))
        if blur % 2 == 0:
            blur += 1
        if sobel not in (1, 3, 5, 7):
            sobel = 3
        sigma = min(10.0, max(0.0, float(self.blur_sigma)))
        x_weight = min(2.0, max(0.0, float(self.gradient_x_weight)))
        y_weight = min(2.0, max(0.0, float(self.gradient_y_weight)))
        if x_weight + y_weight <= 0.0:
            x_weight = y_weight = 1.0
        clip = min(100.0, max(90.0, float(self.clip_percentile)))
        gain = min(3.0, max(0.1, float(self.edge_gain)))
        noise_floor = min(254, max(0, int(self.noise_floor)))
        return SobelConfig(
            blur_ksize=blur,
            blur_sigma=sigma,
            sobel_ksize=sobel,
            gradient_x_weight=x_weight,
            gradient_y_weight=y_weight,
            clip_percentile=clip,
            edge_gain=gain,
            noise_floor=noise_floor,
        )


@dataclass
class TrainReport:
    classes: list[str] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    epochs: int = 0
    train_acc: float = 0.0
    val_acc: float = 0.0
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    seconds: float = 0.0
    device: str = ""
    backbone: str = ""
    resumed: bool = False  # carried on from the previous model for this key

    def text(self) -> str:
        lines = [
            f"backbone      {self.backbone} on {self.device}",
            f"classes       {', '.join(self.classes)}",
            f"images        {self.n_train} train / {self.n_val} validation",
            f"epochs        {self.epochs}   time {self.seconds:.1f}s",
            f"start          {'carried on from the previous model' if self.resumed else 'fresh head'}",
            f"accuracy      train {self.train_acc:.1%}   validation {self.val_acc:.1%}",
        ]
        if self.per_class:
            lines.append("")
            lines.append(f"{'class':<12}{'val n':>7}{'correct':>9}")
            for cls, d in self.per_class.items():
                lines.append(f"{cls:<12}{d['n']:>7}{d['correct']:>9}")
        return "\n".join(lines)


def model_quality_error(report: TrainReport | None) -> str | None:
    """Return why a training report is unsafe for production, or ``None``.

    This gate is independent from checkpoint loading so a caller can show an
    actionable quality result without describing a structurally valid file as
    corrupt.
    """
    if not isinstance(report, TrainReport):
        return "Training report is missing."
    if len(report.classes) != len(MODEL_CLASSES) or set(report.classes) != set(MODEL_CLASSES):
        return "Validation must contain exactly the GOOD and NG classes."
    if isinstance(report.n_val, bool) or not isinstance(report.n_val, int) or report.n_val <= 0:
        return "Validation set is empty."
    try:
        validation_accuracy = float(report.val_acc)
    except (TypeError, ValueError):
        return "Overall validation accuracy is invalid."
    if not math.isfinite(validation_accuracy) or not 0.0 <= validation_accuracy <= 1.0:
        return "Overall validation accuracy is invalid."
    if validation_accuracy < MIN_MODEL_VALIDATION_ACCURACY:
        return (
            f"Overall validation accuracy {validation_accuracy:.1%} is below the "
            f"required {MIN_MODEL_VALIDATION_ACCURACY:.0%}."
        )

    validation_total = 0
    for label in MODEL_CLASSES:
        metrics = report.per_class.get(label) if isinstance(report.per_class, Mapping) else None
        if not isinstance(metrics, Mapping):
            return f"Validation results for {label} are missing."
        count = metrics.get("n")
        correct = metrics.get("correct")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= count
        ):
            return f"Validation results for {label} are invalid."
        validation_total += count
        recall = correct / count
        if recall < MIN_MODEL_CLASS_RECALL:
            return (
                f"Validation recall for {label} {recall:.1%} is below the "
                f"required {MIN_MODEL_CLASS_RECALL:.0%}."
            )
    if validation_total != report.n_val:
        return "Per-class validation counts do not match the validation set size."
    return None


def _model_key(value: object, *, expected: bool = False) -> str:
    if value is None and not expected:
        return ""
    if not isinstance(value, str):
        raise ValueError("Model identity must be text.")
    key = value.strip()
    if expected and not key:
        raise ValueError("Expected model identity cannot be empty.")
    if any(ord(character) < 32 for character in key):
        raise ValueError("Model identity contains control characters.")
    return key


def _checkpoint_report(raw: object, classes: list[str], backbone: str) -> TrainReport | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Model checkpoint report must be an object or null.")
    try:
        report = TrainReport(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model checkpoint report is invalid: {exc}") from exc
    if list(report.classes) != classes:
        raise ValueError("Model checkpoint report classes do not match the model classes.")
    if report.backbone and report.backbone != backbone:
        raise ValueError("Model checkpoint report backbone does not match the model backbone.")
    for name in ("n_train", "n_val", "epochs"):
        value = getattr(report, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Model checkpoint report field {name} is invalid.")
    for name in ("train_acc", "val_acc", "seconds"):
        value = getattr(report, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Model checkpoint report field {name} is invalid.")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"Model checkpoint report field {name} is invalid.")
        if name != "seconds" and number > 1.0:
            raise ValueError(f"Model checkpoint report field {name} is invalid.")
    if not isinstance(report.per_class, Mapping):
        raise ValueError("Model checkpoint per-class report is invalid.")
    for label, metrics in report.per_class.items():
        if label not in classes or not isinstance(metrics, Mapping):
            raise ValueError("Model checkpoint per-class report is invalid.")
        count = metrics.get("n")
        correct = metrics.get("correct")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= count
        ):
            raise ValueError("Model checkpoint per-class report is invalid.")
    return report


def _checkpoint_parts(
    blob: object,
    expected_model_key: str | None = None,
) -> tuple[str, list[str], CropBox, SobelConfig, Mapping, TrainReport | None, str]:
    if not isinstance(blob, Mapping):
        raise ValueError("Model checkpoint must be an object.")

    version = blob.get("preprocess_version")
    if version != PREPROCESS_VERSION:
        raise ValueError(
            f"Model preprocessing is {version or 'legacy colour input'}, expected "
            f"{PREPROCESS_VERSION}. Retrain this model with Sobel input."
        )
    required = {"backbone", "classes", "crop", "sobel", "head_state", "report"}
    missing = sorted(required.difference(blob))
    if missing:
        raise ValueError(f"Model checkpoint is missing: {', '.join(missing)}.")

    backbone = blob["backbone"]
    if not isinstance(backbone, str) or backbone not in BACKBONES:
        raise ValueError(f"Unsupported model backbone: {backbone!r}.")

    raw_classes = blob["classes"]
    if not isinstance(raw_classes, list) or any(not isinstance(item, str) for item in raw_classes):
        raise ValueError("Model checkpoint classes must be a list of names.")
    classes = list(raw_classes)
    if len(classes) != len(MODEL_CLASSES) or set(classes) != set(MODEL_CLASSES):
        raise ValueError("Model checkpoint classes must be exactly GOOD and NG.")

    crop_fields = set(CropBox.__dataclass_fields__)
    raw_crop = blob["crop"]
    if not isinstance(raw_crop, Mapping) or set(raw_crop) != crop_fields:
        raise ValueError("Model checkpoint crop schema is invalid.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_crop.values()):
        raise ValueError("Model checkpoint crop values must be integers.")
    crop = CropBox(**dict(raw_crop))

    sobel_fields = set(SobelConfig.__dataclass_fields__)
    raw_sobel = blob["sobel"]
    if not isinstance(raw_sobel, Mapping) or set(raw_sobel) != sobel_fields:
        raise ValueError("Model checkpoint Sobel schema is invalid.")
    for name in ("blur_ksize", "sobel_ksize", "noise_floor"):
        value = raw_sobel[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Model checkpoint Sobel field {name} must be an integer.")
    for name in (
        "blur_sigma",
        "gradient_x_weight",
        "gradient_y_weight",
        "clip_percentile",
        "edge_gain",
    ):
        value = raw_sobel[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Model checkpoint Sobel field {name} is invalid.")
    sobel_candidate = SobelConfig(**dict(raw_sobel))
    sobel = sobel_candidate.validated()
    if sobel != sobel_candidate:
        raise ValueError("Model checkpoint Sobel values are outside the supported range.")

    state = blob["head_state"]
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise ValueError("Model checkpoint head state schema is invalid.")
    weight, bias = state["weight"], state["bias"]
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("Model checkpoint head state must contain tensors.")
    expected_dimension = BACKBONES[backbone]
    if tuple(weight.shape) != (len(classes), expected_dimension) or tuple(bias.shape) != (
        len(classes),
    ):
        raise ValueError("Model checkpoint head dimensions do not match its backbone and classes.")
    for tensor in (weight, bias):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("Model checkpoint head contains invalid or non-finite tensors.")

    report = _checkpoint_report(blob["report"], classes, backbone)
    persisted_key = _model_key(blob.get("model_key"))
    if expected_model_key is not None:
        expected_key = _model_key(expected_model_key, expected=True)
        if not persisted_key:
            raise ValueError(
                f"Model checkpoint has no identity; expected {expected_key!r}. Retrain or migrate it."
            )
        if persisted_key != expected_key:
            raise ValueError(
                f"Model identity mismatch: checkpoint is {persisted_key!r}, expected {expected_key!r}."
            )
    return backbone, classes, crop, sobel, state, report, persisted_key


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor:
    """Frozen DINOv2 backbone. Loaded lazily so the app starts instantly."""

    def ensure_loaded(self, progress=None) -> None:
        if self._model is not None:
            return
        if progress:
            progress(f"Loading {self.backbone} (first run downloads ~85 MB)...")
        model = torch.hub.load(DINOV2_REPOSITORY, self.backbone, trust_repo=True, verbose=False)
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        if progress:
            progress(f"{self.backbone} ready on {self.device}")

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        device: str | None = None,
        sobel: SobelConfig | None = None,
    ):
        self.backbone = backbone
        self.device = device or pick_device()
        self.dim = BACKBONES.get(backbone, 384)
        self.sobel = (sobel or SobelConfig()).validated()
        self._model = None

    def edge_map(self, image_path: str | Path, crop: CropBox) -> np.ndarray | None:
        """Return the normalized uint8 Sobel magnitude used by the classifier."""
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = crop.apply(img)
        if img.size == 0:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.sobel.blur_ksize > 1:
            gray = cv2.GaussianBlur(
                gray,
                (self.sobel.blur_ksize, self.sobel.blur_ksize),
                self.sobel.blur_sigma,
            )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel.sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.sobel.sobel_ksize)
        gx *= self.sobel.gradient_x_weight
        gy *= self.sobel.gradient_y_weight
        magnitude = cv2.magnitude(gx, gy)
        ceiling = float(np.percentile(magnitude, self.sobel.clip_percentile))
        if not np.isfinite(ceiling) or ceiling <= 1e-6:
            return None
        edge = np.clip(
            magnitude * (255.0 / ceiling) * self.sobel.edge_gain,
            0.0,
            255.0,
        ).astype(np.uint8)
        if self.sobel.noise_floor:
            edge[edge < self.sobel.noise_floor] = 0
        return edge

    def preprocess(self, image_path: str | Path, crop: CropBox) -> np.ndarray | None:
        edge = self.edge_map(image_path, crop)
        if edge is None:
            return None
        edge = cv2.resize(edge, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        rgb = np.repeat(edge[:, :, None], 3, axis=2).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return rgb.transpose(2, 0, 1)

    @torch.no_grad()
    def embed(
        self, paths: list[str], crop: CropBox, batch: int = 16, progress=None, cancelled=None
    ) -> tuple[np.ndarray, list[int]]:
        """Return (embeddings, indices of paths that could be read)."""
        self.ensure_loaded(progress=lambda m: progress(0, len(paths), m) if progress else None)
        out: list[np.ndarray] = []
        kept: list[int] = []
        buffer: list[np.ndarray] = []
        buffer_idx: list[int] = []

        def flush():
            if not buffer:
                return
            x = torch.from_numpy(np.stack(buffer)).to(self.device)
            feats = self._model(x).float().cpu().numpy()
            out.append(feats)
            kept.extend(buffer_idx)
            buffer.clear()
            buffer_idx.clear()

        for i, path in enumerate(paths):
            if cancelled and cancelled():
                raise InterruptedError("Cancelled.")
            arr = self.preprocess(path, crop)
            if arr is not None:
                buffer.append(arr)
                buffer_idx.append(i)
            if len(buffer) >= batch:
                flush()
            if progress and (i + 1) % 8 == 0:
                progress(i + 1, len(paths), "Extracting features")
        flush()
        if progress:
            progress(len(paths), len(paths), "Extracting features")
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32), []
        return np.concatenate(out, axis=0), kept


class CutClassifier:
    """Linear head over DINOv2 features."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        crop: CropBox | None = None,
        sobel: SobelConfig | None = None,
        model_key: str = "",
    ):
        self.backbone = backbone
        self.crop = crop or CropBox()
        self.sobel = (sobel or SobelConfig()).validated()
        self.model_key = _model_key(model_key)
        self.classes: list[str] = []
        self.head: nn.Linear | None = None
        self.device = pick_device()
        self.extractor = FeatureExtractor(backbone, self.device, self.sobel)
        self.report: TrainReport | None = None

    # -- training ----------------------------------------------------------
    def train(
        self,
        items: list[tuple[str, str]],
        epochs: int = 300,
        val_fraction: float = 0.25,
        seed: int = 0,
        progress=None,
        cancelled=None,
        resume_from: "CutClassifier | None" = None,
    ) -> TrainReport:
        """Fit the head on `items`.

        With `resume_from`, the head starts where that model left off instead of
        from random weights, so a round of extra labels refines what is already
        there. It is only carried on when the two models match - same backbone,
        same classes, same feature size - otherwise the weights would be
        meaningless and a fresh head is used.
        """
        counts = {
            label: sum(item_label == label for _, item_label in items) for label in MODEL_CLASSES
        }
        unexpected = sorted({label for _, label in items}.difference(MODEL_CLASSES))
        if unexpected:
            raise ValueError(
                f"Training labels must be exactly GOOD and NG; found {', '.join(unexpected)}."
            )
        short = [label for label in MODEL_CLASSES if counts[label] < 2]
        if short:
            detail = ", ".join(f"{label}={counts[label]}" for label in MODEL_CLASSES)
            raise ValueError(
                f"Need at least 2 labelled images per class for train/validation split; {detail}."
            )
        started = time.time()
        paths = [p for p, _ in items]
        names = sorted({c for _, c in items})

        feats, kept = self.extractor.embed(paths, self.crop, progress=progress, cancelled=cancelled)
        expected_indices = list(range(len(paths)))
        if kept != expected_indices:
            unreadable = len(paths) - len(set(kept).intersection(expected_indices))
            raise ValueError(
                f"{max(1, unreadable)} labelled image(s) could not be read; training aborted."
            )
        labels = np.array([names.index(items[i][1]) for i in kept], dtype=np.int64)

        # stratified split so every class appears in validation where possible
        rng = np.random.default_rng(seed)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for c in range(len(names)):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())
        if not train_idx:
            raise ValueError("No training samples left after the split.")

        x = torch.from_numpy(feats).float().to(self.device)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.from_numpy(labels).to(self.device)
        xt, yt = x[train_idx], y[train_idx]
        xv, yv = x[val_idx], y[val_idx]

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = nn.Linear(feats.shape[1], len(names)).to(self.device)
        resumed = False
        if (
            resume_from is not None
            and resume_from.head is not None
            and list(resume_from.classes) == names
            and resume_from.backbone == self.backbone
            and resume_from.head.in_features == feats.shape[1]
            and resume_from.head.out_features == len(names)
        ):
            head.load_state_dict(resume_from.head.state_dict())
            resumed = True

        # class weights keep a rare defect class from being ignored
        counts = np.bincount(labels[train_idx], minlength=len(names)).astype(np.float32)
        weights = (
            torch.from_numpy(np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0))
            .float()
            .to(self.device)
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

        for epoch in range(epochs):
            if cancelled and cancelled():
                raise InterruptedError("Cancelled.")
            head.train()
            opt.zero_grad()
            loss = loss_fn(head(xt), yt)
            loss.backward()
            opt.step()
            if progress and (epoch + 1) % 50 == 0:
                progress(epoch + 1, epochs, f"Training head (loss {loss.item():.3f})")

        head.eval()
        with torch.no_grad():
            train_acc = (head(xt).argmax(1) == yt).float().mean().item()
            if len(val_idx):
                pred_v = head(xv).argmax(1)
                val_acc = (pred_v == yv).float().mean().item()
                per_class = {}
                for c, name in enumerate(names):
                    mask = yv == c
                    n = int(mask.sum().item())
                    if n:
                        per_class[name] = {
                            "n": n,
                            "correct": int((pred_v[mask] == c).sum().item()),
                        }
            else:
                val_acc, per_class = float("nan"), {}

        self.classes = names
        self.head = head
        self.report = TrainReport(
            classes=names,
            n_train=len(train_idx),
            n_val=len(val_idx),
            epochs=epochs,
            train_acc=train_acc,
            val_acc=val_acc,
            per_class=per_class,
            seconds=time.time() - started,
            device=self.device,
            backbone=self.backbone,
            resumed=resumed,
        )
        return self.report

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, paths: list[str], progress=None, cancelled=None) -> list[tuple[str, float]]:
        if self.head is None:
            raise RuntimeError("No trained model loaded.")
        feats, kept = self.extractor.embed(paths, self.crop, progress=progress, cancelled=cancelled)
        results: list[tuple[str, float]] = [("", 0.0)] * len(paths)
        if not kept:
            return results
        x = torch.nn.functional.normalize(torch.from_numpy(feats).float().to(self.device), dim=1)
        probs = torch.softmax(self.head(x), dim=1).cpu().numpy()
        for slot, i in enumerate(kept):
            c = int(probs[slot].argmax())
            results[i] = (self.classes[c], float(probs[slot][c]))
        return results

    def predict_one(self, path: str) -> tuple[str, float]:
        return self.predict([path])[0]

    # -- persistence -------------------------------------------------------
    def save(
        self,
        path: Path,
        protected: list[Path] | None = None,
        model_key: str | None = None,
    ) -> None:
        if self.head is None:
            raise RuntimeError("Nothing to save - train a model first.")
        path = Path(path)
        sidecar = path.with_suffix(".json")
        if sidecar == path:
            raise ValueError("Model path must not use the .json sidecar extension.")
        roots = list(protected or ())
        check_write_target(path, roots)
        check_write_target(sidecar, roots)

        identity = self.model_key if model_key is None else _model_key(model_key)
        metadata = {
            "preprocess_version": PREPROCESS_VERSION,
            "backbone": self.backbone,
            "classes": self.classes,
            "crop": asdict(self.crop),
            "sobel": asdict(self.sobel),
            "report": asdict(self.report) if self.report else None,
            "model_key": identity,
        }
        payload = {**metadata, "head_state": self.head.state_dict()}
        _checkpoint_parts(payload)

        path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        model_temporary = path.with_name(f".{path.name}.{nonce}.tmp")
        sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{nonce}.tmp")
        check_write_target(model_temporary, roots)
        check_write_target(sidecar_temporary, roots)
        try:
            torch.save(payload, model_temporary)
            sidecar_temporary.write_text(
                json.dumps(metadata, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            model_temporary.replace(path)
            sidecar_temporary.replace(sidecar)
        finally:
            for temporary in (model_temporary, sidecar_temporary):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def load(
        cls,
        path: Path,
        expected_model_key: str | None = None,
    ) -> "CutClassifier":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        backbone, classes, crop, sobel, state, report, model_key = _checkpoint_parts(
            blob, expected_model_key
        )
        model = cls(backbone=backbone, crop=crop, sobel=sobel, model_key=model_key)
        model.classes = classes
        head = nn.Linear(BACKBONES[model.backbone], len(model.classes))
        head.load_state_dict(dict(state), strict=True)
        model.head = head.eval().to(model.device)
        model.report = report
        return model
