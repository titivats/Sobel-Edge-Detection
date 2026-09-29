"""Standalone reference-edge experiment; does not change production settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import read_pixel_size
from .guard import ProtectedPathError, check_write_target
from .machine_reference import load_manifest, reference_for_image
from .reference import image_signature, measure_reference, normal_scale


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def side_from_point(line, point) -> int:
    start, end = np.asarray(line, dtype=float)
    vector = end - start
    length = np.linalg.norm(vector)
    if length < 12:
        raise ValueError("Reference must be at least 12 original pixels long.")
    normal = np.array([-vector[1], vector[0]]) / length
    distance = float(np.dot(np.asarray(point) - start, normal))
    if abs(distance) < 2:
        raise ValueError("Click inside the PCB, farther from the reference line.")
    return 1 if distance > 0 else -1


def pick_reference(image):
    """Three-click experiment window; all coordinates return to original pixels."""
    h, w = image.shape[:2]
    ratio = min(1000 / w, 700 / h, 1.0)
    view_w, view_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    sx, sy = view_w / w, view_h / h
    base = cv2.resize(image, (view_w, view_h), interpolation=cv2.INTER_AREA)
    clicks = []
    title = "Edge trial - 2 reference endpoints, then PCB side"

    def clicked(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append([x / sx, y / sy])

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, clicked)
    try:
        while True:
            shown = base.copy()
            for index, point in enumerate(clicks):
                pos = (round(point[0] * sx), round(point[1] * sy))
                cv2.circle(shown, pos, 5, (255, 0, 255) if index == 2 else (255, 180, 0), -1)
            if len(clicks) >= 2:
                a, b = [(round(p[0] * sx), round(p[1] * sy)) for p in clicks[:2]]
                cv2.line(shown, a, b, (255, 180, 0), 2)
            instruction = (
                "Click START of nominal edge",
                "Click END of nominal edge",
                "Click inside PCB material",
                "ENTER: measure | R: reset | ESC: cancel",
            )[len(clicks)]
            cv2.rectangle(shown, (0, 0), (view_w, 40), (20, 20, 20), -1)
            cv2.putText(
                shown,
                instruction,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(title, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (ord("r"), ord("R")):
                clicks.clear()
            if key in (10, 13) and len(clicks) == 3:
                try:
                    return clicks[:2], side_from_point(clicks[:2], clicks[2])
                except ValueError as exc:
                    print(exc)
                    clicks.clear()
    finally:
        cv2.destroyAllWindows()


def run_trial(
    image_path,
    line,
    *,
    side=1,
    scales,
    radius=45,
    contrast=3.0,
    polarity=1,
    output_root,
    scale_source="manual",
    eqp_path=None,
    reference_note="Operator-selected nominal edge",
    expected_source=None,
    machine_reference=None,
) -> Path:
    path = Path(image_path).resolve()
    output_root = Path(output_root).resolve()
    # Protect the full export when the image belongs to Picture, including subfolders.
    picture = next((p for p in path.parents if p.name.casefold() == "picture"), None)
    roots = [picture.parent if picture else path.parent]
    if machine_reference:
        roots.extend(Path(s["path"]).resolve().parent for s in machine_reference["source_files"])
    if eqp_path:
        cfg = Path(eqp_path).resolve()
        roots.append(cfg.parent.parent if cfg.parent.name.casefold() == "config" else cfg.parent)
    check_write_target(output_root, roots)
    # Validate physical scale even if no boundary can be measured.
    normal_scale([0, 1], *scales)
    source = image_signature(path)
    if expected_source is not None and source != expected_source:
        raise ValueError("Source image changed while selecting the reference. Reopen the image.")
    image = read_image(path)
    profile = measure_reference(
        image, line, side=side, radius=radius, min_contrast=contrast, polarity=polarity
    )
    if not len(profile.anchors):
        raise ValueError(profile.reason)
    factor = normal_scale(profile.normal, *scales)
    complete = profile.valid
    summary = {
        "version": 1,
        "image": str(path),
        "source_signature": source,
        "original_size_px": [image.shape[1], image.shape[0]],
        "reference_line_px": line,
        "reference_note": reference_note,
        "pcb_side": side,
        "search_radius_px": radius,
        "min_contrast": contrast,
        "polarity": polarity,
        "scale_mm_per_px_xy": list(scales),
        "normal_mm_per_px": factor,
        "scale_source": scale_source,
        "calibration_verified": False,
        "measurement_status": "COMPLETE_ESTIMATE" if complete else "INCOMPLETE",
        "reason": profile.reason,
        "coverage": profile.coverage,
        "inward_mm": profile.inward_px * factor if complete else None,
        "protrusion_mm": profile.protrusion_px * factor if complete else None,
        "accepted_samples_only": {
            "inward_mm": profile.inward_px * factor if profile.accepted.any() else None,
            "protrusion_mm": profile.protrusion_px * factor if profile.accepted.any() else None,
        },
        "production_decision": None,
    }
    if machine_reference is not None:
        summary["machine_reference"] = machine_reference
    # Yellow is the protruding boundary (negative signed displacement).
    # Report observed maxima on both sides, even when coverage is partial.
    observed_max = profile.protrusion_px * factor if profile.accepted.any() else None
    yellow_indices = np.flatnonzero(profile.accepted & (profile.deviations < 0))
    maximum_index = (
        int(yellow_indices[np.argmin(profile.deviations[yellow_indices])])
        if len(yellow_indices)
        else None
    )
    summary["max_blue_yellow_mm"] = observed_max if complete else None
    summary["observed_max_blue_yellow_mm"] = observed_max
    summary["max_edge_point_px"] = (
        profile.points[maximum_index].tolist() if maximum_index is not None else None
    )
    inner_indices = np.flatnonzero(profile.accepted & (profile.deviations > 0))
    inner_index = (
        int(inner_indices[np.argmax(profile.deviations[inner_indices])])
        if len(inner_indices)
        else None
    )
    observed_inner = summary["accepted_samples_only"]["inward_mm"]
    summary["inner_line_max_mm"] = observed_inner
    summary["outer_line_max_mm"] = observed_max
    summary["line_max_scope"] = "accepted_samples_only"
    summary["inner_line_max_point_px"] = (
        profile.points[inner_index].tolist() if inner_index is not None else None
    )
    summary["outer_line_max_point_px"] = summary["max_edge_point_px"]
    if source != image_signature(path):
        raise ValueError("Source image changed during measurement. Retry with a stable image.")
    destination = output_root / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    )
    check_write_target(destination, roots)
    destination.mkdir(parents=True, exist_ok=False)
    annotated = image.copy()
    ends = np.rint(line).astype(int)
    cv2.line(annotated, tuple(ends[0]), tuple(ends[1]), (255, 180, 0), 2)
    midpoint = np.mean(ends, axis=0)
    cv2.arrowedLine(
        annotated,
        tuple(midpoint.astype(int)),
        tuple((midpoint + profile.normal * 50).astype(int)),
        (255, 0, 255),
        2,
    )
    with (destination / "profile.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reference_x_px",
                "reference_y_px",
                "edge_x_px",
                "edge_y_px",
                "accepted",
                "signed_deviation_px",
                "estimated_signed_mm",
            ]
        )
        for anchor, point, delta, accepted in zip(
            profile.anchors, profile.points, profile.deviations, profile.accepted
        ):
            writer.writerow(
                [
                    *anchor,
                    *(point if accepted else ["", ""]),
                    bool(accepted),
                    float(delta) if accepted else "",
                    float(delta * factor) if accepted else "",
                ]
            )
            position = point if accepted else anchor
            colour = (0, 0, 255) if delta > 0 else (0, 255, 255)
            cv2.circle(
                annotated,
                tuple(np.rint(position).astype(int)),
                2,
                colour if accepted else (160, 160, 160),
                -1,
            )
    # A separate banner preserves the original image coordinate system in CSV.
    banner = np.full((115, image.shape[1], 3), 24, dtype=np.uint8)

    def number(value):
        return "N/A" if value is None else f"{value:.4f} mm"

    for label, index, colour in (
        ("inner line", inner_index, (0, 100, 255)),
        ("outer line", maximum_index, (0, 255, 255)),
    ):
        if index is None:
            continue
        edge = profile.points[index]
        physical_normal = profile.normal / np.asarray(scales)
        physical_normal /= np.linalg.norm(physical_normal)
        signed_mm = float(profile.deviations[index]) * factor
        foot = edge - signed_mm * physical_normal / np.asarray(scales)
        a, b = tuple(np.rint(foot).astype(int)), tuple(np.rint(edge).astype(int))
        cv2.line(annotated, a, b, (255, 255, 255), 2)
        cv2.circle(annotated, b, 7, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"{label} MAX {abs(signed_mm):.4f} mm",
            (max(0, b[0] + 12), max(20, b[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )
    lines = [
        f"inner line MAX: {number(observed_inner)} | outer line MAX: {number(observed_max)}"
        + ("" if complete else " | PARTIAL - accepted samples only"),
        f"Coverage {profile.coverage:.1%} | ESTIMATED - scale and nominal edge must be verified",
        "Blue: reference | Red: inner line | Yellow: outer line | White: MAX | Gray: rejected",
    ]
    font_scale = min(0.65, max(0.25, image.shape[1] / 2100))
    for index, text in enumerate(lines):
        cv2.putText(
            banner,
            text,
            (10, 28 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    canvas = np.vstack([banner, annotated])
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError("Could not encode measurement image.")
    encoded.tofile(destination / "annotated.png")
    (destination / "measurement.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--browse-dir", default="")
    parser.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--side", type=int, choices=(-1, 1))
    parser.add_argument("--machine-reference", type=Path, help="Image-bound audit manifest JSON")
    parser.add_argument("--edge", choices=("A", "B"), default="B")
    parser.add_argument("--scale", nargs=2, type=float, metavar=("X_MM_PX", "Y_MM_PX"))
    parser.add_argument("--eqp-config", type=Path)
    parser.add_argument("--radius", type=int, default=45)
    parser.add_argument("--contrast", type=float, default=3)
    parser.add_argument("--polarity", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--reference-note", default="Operator-selected nominal edge")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "measurement_trials")
    args = parser.parse_args(argv)
    interactive_image = args.image is None
    try:
        if args.machine_reference and any(
            v is not None for v in (args.line, args.side, args.scale, args.eqp_config)
        ):
            raise ValueError("Machine reference cannot be combined with manual line / side / scale")
        if args.image is None:
            from PySide6.QtWidgets import QApplication, QFileDialog

            app = QApplication.instance() or QApplication([])
            chosen, _ = QFileDialog.getOpenFileName(
                None,
                "Choose original PCB image",
                args.browse_dir,
                "Images (*.bmp *.png *.jpg *.jpeg)",
            )
            if not chosen:
                return 0
            args.image = Path(chosen)
            del app
        picture = next(
            (p for p in args.image.resolve().parents if p.name.casefold() == "picture"), None
        )
        config = args.eqp_config or (picture.parent / "Config" / "eqp.cfg" if picture else None)
        selected_source = image_signature(args.image)
        recorded = None
        if args.machine_reference:
            recorded = reference_for_image(
                load_manifest(args.machine_reference), args.image, args.edge
            )
            scales, scale_source = recorded["scales"], str(args.machine_reference.resolve())
            config = None
        elif args.scale:
            scales, scale_source = args.scale, "Manual unverified X/Y scale"
        elif config and config.is_file():
            scales, scale_source = read_pixel_size(config), str(config.resolve())
        else:
            raise ValueError("No PixelSize found. Supply --eqp-config FILE or --scale X Y.")
        if not all(math.isfinite(v) and v > 0 for v in scales):
            raise ValueError("Scale must be positive and finite.")
        if recorded:
            line, side = recorded["line"], recorded["side"]
        elif args.line:
            line, side = [args.line[:2], args.line[2:]], args.side or 1
        else:
            picked = pick_reference(read_image(args.image))
            if picked is None:
                return 0
            line, side = picked
        destination = run_trial(
            args.image,
            line,
            side=side,
            scales=scales,
            radius=args.radius,
            contrast=args.contrast,
            polarity=args.polarity,
            output_root=args.output,
            scale_source=scale_source,
            eqp_path=config,
            reference_note=(
                f"Recorded-alignment router tangent {args.edge}; conditional commanded camera pose"
                if recorded
                else args.reference_note
            ),
            expected_source=selected_source,
            machine_reference=recorded["provenance"] if recorded else None,
        )
        print(
            f"Trial saved: {destination}\nOpen annotated.png; details: measurement.json / profile.csv"
        )
        if interactive_image or (args.line is None and recorded is None):
            preview = read_image(destination / "annotated.png")
            height, width = preview.shape[:2]
            ratio = min(1100 / width, 800 / height, 1)
            cv2.imshow(
                "Measurement result - press any key to close",
                cv2.resize(preview, (round(width * ratio), round(height * ratio))),
            )
            while cv2.waitKey(30) < 0:
                if (
                    cv2.getWindowProperty(
                        "Measurement result - press any key to close", cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    break
            cv2.destroyAllWindows()
        return 0
    except (OSError, ValueError, KeyError, TypeError, ProtectedPathError, cv2.error) as exc:
        print(f"Measurement not completed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
