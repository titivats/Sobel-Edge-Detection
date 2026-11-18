"""Generate installer and uninstaller instructions for the exact portable payload."""

import sys
from pathlib import Path


def literal(value):
    # Escape a literal dollar in an NSIS instruction. Windows names cannot contain quotes.
    return str(value).replace("$", "$$")


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    destination = Path(sys.argv[2])
    for required in ("AVTR.exe", "production_app.py", "config.json"):
        if not (root / required).is_file():
            raise SystemExit(f"Portable payload is missing {required}")
    for required in ("_runtime", "router_vision", "models"):
        if not (root / required).is_dir():
            raise SystemExit(f"Portable payload is missing {required}")

    state_files = {
        "config.json",
        "labels.json",
        "baselines.json",
        "sobel_workflow.json",
        "positions.json",
        "reference_lines.json",
    }
    instructions = []
    managed_files = []
    managed_dirs = set()
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    for seeds in (False, True):
        previous = None
        instructions.append("SetOverwrite off" if seeds else "SetOverwrite on")
        for path in files:
            is_seed = path.parts[0].lower() == "models" or str(path).lower() in state_files
            if is_seed != seeds:
                continue
            if path.parent != previous:
                target = (
                    "$INSTDIR" if path.parent == Path(".") else "$INSTDIR\\" + literal(path.parent)
                )
                instructions.append(f'SetOutPath "{target}"')
                previous = path.parent
            instructions.append(f'File "{literal(root / path)}"')
            if not is_seed:
                managed_files.append(path)
                managed_dirs.update(p for p in path.parents if p != Path("."))

    lines = [
        "!macro InstallPortableFiles",
        *instructions,
        "SetOverwrite on",
        "!macroend",
        "!macro UninstallPortableFiles",
    ]
    for path in managed_files:
        lines.append(f'Delete "$INSTDIR\\{literal(path)}"')
    for folder in sorted(managed_dirs, key=lambda d: (-len(d.parts), str(d))):
        lines.append(f'RMDir "$INSTDIR\\{literal(folder)}"')
    lines.append("!macroend")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
