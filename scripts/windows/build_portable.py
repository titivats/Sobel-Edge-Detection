"""Build a relocatable Windows x64 AVTR from the tested interpreter and installed wheels."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DINO_REPO = "facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS = "dinov2_vits14_pretrain.pth"
WEIGHTS_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
PACKAGES = (
    "PySide6",
    "PySide6-Essentials",
    "shiboken6",
    "numpy",
    "opencv-python",
    "torch",
    "filelock",
    "fsspec",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "setuptools",
    "sympy",
    "typing-extensions",
)
QT_FILES = {
    "QtCore.pyd",
    "QtGui.pyd",
    "QtWidgets.pyd",
    "Qt6Core.dll",
    "Qt6Gui.dll",
    "Qt6Widgets.dll",
    "Qt6Svg.dll",
    "pyside6.abi3.dll",
    "opengl32sw.dll",
}
QT_PLUGINS = {"platforms", "styles", "imageformats", "iconengines"}


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def include_wheel_file(path):
    if ".." in path.parts or "__pycache__" in path.parts or path.suffix in {".pyc", ".pdb", ".lib"}:
        return False
    if path.parts[:2] == ("torch", "include"):
        return False
    if path.parts[0] != "PySide6":
        return True
    if len(path.parts) == 2:
        return (
            path.suffix == ".py"
            or path.name in QT_FILES
            or path.name.startswith(("vcruntime", "msvcp140", "concrt140"))
        )
    return path.parts[1] == "support" or (
        path.parts[1] == "plugins" and path.parts[2] in QT_PLUGINS
    )


def copy_vc_runtime(cabinet, runtime):
    # Use the x64 VC runtime redistributable. The Qt wheel omits atomic_wait, which Torch uses.
    with tempfile.TemporaryDirectory(prefix="avtr-vcredist-") as directory:
        subprocess.run(
            [
                str(Path(os.environ["WINDIR"]) / "System32/expand.exe"),
                str(cabinet),
                "-F:*.dll_amd64",
                directory,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        extracted = Path(directory)
        for name in (
            "concrt140",
            "msvcp140",
            "msvcp140_1",
            "msvcp140_2",
            "msvcp140_atomic_wait",
            "msvcp140_codecvt_ids",
            "vcruntime140",
            "vcruntime140_1",
            "vcruntime140_threads",
        ):
            copy(extracted / f"{name}.dll_amd64", runtime / f"{name}.dll")


def copy_runtime(runtime, vc_cab):
    if sys.platform != "win32" or struct.calcsize("P") != 8 or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Build with the project's Windows x64 Python 3.12 environment.")
    base = Path(sys.base_prefix)
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python312.dll", "LICENSE.txt"):
        copy(base / name, runtime / name)
    shutil.copytree(base / "DLLs", runtime / "DLLs")
    shutil.copytree(
        base / "Lib",
        runtime / "Lib",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "site-packages",
            "test",
            "tests",
        ),
    )
    site_packages = runtime / "Lib" / "site-packages"
    for name in PACKAGES:
        dist = metadata.distribution(name)
        for entry in dist.files:
            relative = Path(entry)
            if include_wheel_file(relative):
                source = Path(dist.locate_file(entry))
                if source.is_file():
                    copy(source, site_packages / relative)
    copy_vc_runtime(vc_cab, runtime)
    (runtime / "python312._pth").write_text(
        ".\nDLLs\nLib\nLib\\site-packages\n..\nimport site\n", encoding="ascii"
    )


def copy_backbone(hub, runtime):
    source = hub / DINO_REPO
    target_hub = runtime / "torch" / "hub"
    target = target_hub / DINO_REPO
    for name in ("hubconf.py", "LICENSE", "MODEL_CARD.md", "dinov2/__init__.py"):
        copy(source / name, target / name)
    for name in ("hub", "layers", "models"):
        for module in (source / "dinov2" / name).rglob("*.py"):
            copy(module, target / module.relative_to(source))
    checkpoint = hub / "checkpoints" / WEIGHTS
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA:
            raise ValueError("The pinned DINOv2 S/14 checkpoint does not match.")
    copy(checkpoint, target_hub / "checkpoints" / WEIGHTS)
    (target_hub / "trusted_list").write_text("facebookresearch_dinov2\n", encoding="ascii")


def copy_app(app, target, *, public=False):
    copy(app / "production_app.py", target / "production_app.py")
    for module in (app / "router_vision").glob("*.py"):
        if module.name not in {"measurement_trial.py", "fiducial_projection.py"}:
            copy(module, target / "router_vision" / module.name)
    (target / "models").mkdir(exist_ok=True)
    if not public:
        for model in (app / "models").glob("cut_classifier*"):
            if model.suffix in {".json", ".pt"}:
                copy(model, target / "models" / model.name)
    config = json.loads((app / "config.example.json").read_text(encoding="utf-8"))
    local_path = app / "config.json"
    if not public and local_path.is_file():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        for key in config:
            if key in local:
                config[key] = local[key]
    config.update(
        picture_dir="SepData/Picture",
        result_dir="SepData/Result",
        recipe_dir="SepData/Recipe",
        eqp_cfg_path="SepData/Config/Eqp.cfg",
        model_dir="",
        capture_manifest_dir="",
        active_recipe="",
        active_program_key="",
        active_program_recipe="",
        active_model_sha256="",
    )
    (target / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    copy(HERE / "README-TH.txt", target / "README-TH.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--public", action="store_true", help="Package example settings and an empty models folder."
    )
    parser.add_argument(
        "--vc-cab",
        required=True,
        type=Path,
        help="cab1.cab from Microsoft's x64 vcRuntimeMinimum redistributable",
    )
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache/torch/hub")
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise ValueError("Choose a new empty output directory to preserve existing user data.")
    runtime = target / "_runtime"
    copy_runtime(runtime, args.vc_cab.resolve())
    copy_backbone(args.hub.resolve(), runtime)
    copy_app(args.app.resolve(), target, public=args.public)
    shutil.copytree(HERE / "licenses", runtime / "licenses")
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    subprocess.run(
        [
            str(compiler),
            "/nologo",
            "/target:winexe",
            "/platform:x64",
            "/codepage:65001",
            "/reference:System.Windows.Forms.dll",
            "/reference:System.Drawing.dll",
            "/reference:System.Security.dll",
            f"/out:{target / 'AVTR.exe'}",
            str(HERE / "Launcher.cs"),
        ],
        check=True,
    )
    print(target / "AVTR.exe")


if __name__ == "__main__":
    main()
