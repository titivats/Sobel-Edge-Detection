"""Select one program after assigning images against the complete Result archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .auo6000 import AUO6000Dataset


def recipe_name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


@dataclass(frozen=True)
class ProgramSelection:
    product: str
    recipe: str
    table: str = ""

    @property
    def route(self) -> str:
        return f"{self.product}|{self.table}" if self.table else self.product

    @property
    def key(self) -> str:
        identity = json.dumps([self.product, recipe_name(self.recipe), self.table])
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        return f"{self.product or 'recipe'}__program_{digest}"

    @property
    def caption(self) -> str:
        details = " | ".join(value for value in (self.product, self.table) if value)
        return f"{self.recipe} | {details}" if details else self.recipe

    def matches(self, panel) -> bool:
        return (
            panel.product_id.strip() == self.product
            and recipe_name(panel.recipe_name) == recipe_name(self.recipe)
            and panel.table.strip() == self.table
        )


def available_programs(dataset: AUO6000Dataset) -> dict[str, ProgramSelection]:
    programs = {}
    for panel in dataset.panels:
        if panel.product_id.strip() and panel.recipe_name.strip():
            program = ProgramSelection(
                panel.product_id.strip(), panel.recipe_name.strip(), panel.table.strip()
            )
            programs.setdefault(program.key, program)
    known = {recipe_name(program.recipe) for program in programs.values()}
    for path in dataset.recipe_files:
        if recipe_name(Path(path).name) not in known:
            program = ProgramSelection("", Path(path).name)
            programs.setdefault(program.key, program)
    return dict(sorted(programs.items(), key=lambda item: item[1].caption.casefold()))


def select_program(dataset: AUO6000Dataset, program: ProgramSelection) -> AUO6000Dataset:
    panels = [panel for panel in dataset.panels if program.matches(panel)]
    result_files = {panel.result_file for panel in panels}
    images = [image for image in dataset.images if image.result_file in result_files]
    known = [p for p in available_programs(dataset).values() if p.product]
    # For a single known program, keep unmatched images visible as UNASSIGNED.
    # Never claim them for a program or renumber cuts after filtering.
    if len(known) == 1 and known[0].key == program.key:
        images = list(dataset.images)
    warnings = []
    unmatched = sum(not image.result_file for image in dataset.images)
    if unmatched:
        warnings.append(
            f"{unmatched} source image(s) have no verified Result assignment. "
            "Unassigned images cannot be used as another program's training data."
        )
    if not panels:
        warnings.append("No Result records identify this selected recipe and product.")
    if not images:
        warnings.append("No matched images are available for the selected program.")
    if any(Path(i.path).suffix.casefold() == ".bmp" and i.stored_format != "BMP" for i in images):
        warnings.append("Some .bmp filenames contain another image format; decoded by content.")
    return replace(
        dataset,
        panels=panels,
        images=images,
        recipe_files=[
            p for p in dataset.recipe_files if recipe_name(p) == recipe_name(program.recipe)
        ],
        warnings=warnings,
    )
