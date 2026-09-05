from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from router_vision.auo6000 import (
    AUO6000Dataset,
    AUO6000Image,
    AUO6000Panel,
    scan_auo6000_dataset,
)


CSV_HEADER = (
    "SN,Barcode,Recipe_Name,ProductId,BitDiameter,CuttingSpeed,"
    "OffsetY,OffsetX,Result,BitShiftCount,CuttingTime\n"
)

LEGACY_CSV_HEADER = (
    "SN,ID,Barcode,ProductId,CutedTable,Recipe_Name,Result,OffsetX,OffsetY,"
    "RotateAngle,Start_time,Tact_time,OperatorID,Length,Width,ConveyorSpeed,"
    "CuttingTime,Message,MachineID,SubBoardCount,End_time,FixtureBarcode\n"
)


class AUO6000DatasetTests(unittest.TestCase):
    @staticmethod
    def _folders(root: Path) -> tuple[Path, Path, Path, Path]:
        folders = tuple(root / name for name in ("Picture", "Result", "Recipe", "Log"))
        for path in folders:
            path.mkdir()
        return folders

    @staticmethod
    def _image(path: Path) -> None:
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")

    def test_links_timestamped_picture_groups_to_panel_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture, result, recipe, log = self._folders(root)

            png_signature = b"\x89PNG\r\n\x1a\n" + b"test"
            for name in (
                "20260903_111059.bmp",
                "20260903_111104.bmp",
                "20260903_111327.bmp",
                "20260903_111332.bmp",
            ):
                (picture / name).write_bytes(png_signature)

            (result / "_20260903_111247.csv").write_text(
                CSV_HEADER
                + "53391,,199944000-Optoput-Test.rcp,199944000-Optoput-Test,"
                "1.3,4,0.03,-0.08,True,0,89\n"
                + ",,199944000-Optoput-Test.rcp,199944000-Optoput-Test,"
                "1.3,4,0,0,True,0,-1\n",
                encoding="utf-8",
            )
            (result / "_20260903_111515.csv").write_text(
                CSV_HEADER
                + "53392,,199944000-Optoput-Test.rcp,199944000-Optoput-Test,"
                "1.3,4,-0.08,0.001,True,0,89\n",
                encoding="utf-8",
            )
            (recipe / "199944000-Optoput-Test.rcp").write_bytes(b"recipe")
            (log / "Sep_20260903.log").write_text("log", encoding="utf-8")

            dataset = scan_auo6000_dataset(picture)

            self.assertEqual(Path(dataset.root), root)
            self.assertEqual(len(dataset.panels), 2)
            self.assertEqual(len(dataset.images), 4)
            self.assertEqual(dataset.expected_cut_points, 2)
            self.assertEqual(dataset.recipe_name, "199944000-Optoput-Test.rcp")
            self.assertEqual(dataset.format_summary, "PNG 4")
            self.assertEqual(
                [(image.panel_sn, image.cut_point) for image in dataset.images],
                [("53391", 1), ("53391", 2), ("53392", 1), ("53392", 2)],
            )
            self.assertTrue(all(image.machine_result == "GOOD" for image in dataset.images))
            self.assertIn(".bmp filenames", dataset.warnings[0])

    def test_legacy_time_windows_split_back_to_back_panels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture, result, _recipe, _log = self._folders(root)
            for stamp in ("100000", "100005", "100010", "100015"):
                self._image(picture / f"20260903_{stamp}.bmp")

            rows = (
                (
                    "_20260903_100006.csv",
                    "A1,id-1,,PRODUCT-A,LeftTable,PRODUCT-A.rcp,True,0,0,0,"
                    "2026/09/03 10:00:00,6,OP,0,0,0,6,,AUO6000,1,"
                    "2026/09/03 10:00:06,\n",
                ),
                (
                    "_20260903_100015.csv",
                    "A2,id-2,,PRODUCT-A,LeftTable,PRODUCT-A.rcp,True,0,0,0,"
                    "2026/09/03 10:00:07,8,OP,0,0,0,8,,AUO6000,1,"
                    "2026/09/03 10:00:15,\n",
                ),
            )
            for name, row in rows:
                (result / name).write_text(LEGACY_CSV_HEADER + row, encoding="utf-8")

            dataset = scan_auo6000_dataset(root)

            self.assertEqual(
                [(image.panel_sn, image.cut_point, image.cut_point_total)
                 for image in dataset.images],
                [("A1", 1, 2), ("A1", 2, 2), ("A2", 1, 2), ("A2", 2, 2)],
            )
            self.assertEqual(dataset.panels[0].start_at.strftime("%H:%M:%S"), "10:00:00")
            self.assertEqual(dataset.panels[1].end_at.strftime("%H:%M:%S"), "10:00:15")
            self.assertFalse(any("did not receive" in warning for warning in dataset.warnings))

    def test_modern_pairing_is_bounded_and_reports_both_unmatched_sides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture, result, _recipe, _log = self._folders(root)
            self._image(picture / "20260903_100000.bmp")
            (result / "_20260903_100500.csv").write_text(
                CSV_HEADER
                + "P1,,PRODUCT-A.rcp,PRODUCT-A,1.3,4,0,0,True,0,10\n",
                encoding="utf-8",
            )

            dataset = scan_auo6000_dataset(root)

            self.assertEqual(dataset.images[0].panel_sn, "")
            self.assertTrue(any("1 image(s) could not be linked" in item
                                for item in dataset.warnings))
            self.assertTrue(any("1 panel result(s) did not receive" in item
                                for item in dataset.warnings))

    def test_modern_result_boundary_splits_back_to_back_panels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture, result, _recipe, _log = self._folders(root)
            for stamp in ("100000", "100005", "100015", "100020"):
                self._image(picture / f"20260903_{stamp}.bmp")
            for stamp, sn in (("100010", "P1"), ("100025", "P2")):
                (result / f"_20260903_{stamp}.csv").write_text(
                    CSV_HEADER + f"{sn},,PRODUCT-A.rcp,PRODUCT-A,1.3,4,0,0,True,0,10\n",
                    encoding="utf-8",
                )
            dataset = scan_auo6000_dataset(root)
            self.assertEqual([image.panel_sn for image in dataset.images], ["P1", "P1", "P2", "P2"])
            self.assertEqual(dataset.expected_cut_points, 2)

    def test_expected_cut_points_counts_each_group_once(self):
        images = []
        for group, total in enumerate((2, 2, 5)):
            for cut_point in range(1, total + 1):
                images.append(
                    AUO6000Image(
                        path=f"group-{group}-cut-{cut_point}.bmp",
                        captured_at=None,
                        cut_point=cut_point,
                        cut_point_total=total,
                    )
                )
        self.assertEqual(AUO6000Dataset(root="", picture_dir="", images=images).expected_cut_points, 2)

    def test_expected_cut_points_is_ambiguous_across_products(self):
        panels = [
            AUO6000Panel("A", "GOOD", "a.csv", None, product_id="PRODUCT-A"),
            AUO6000Panel("B", "GOOD", "b.csv", None, product_id="PRODUCT-B"),
        ]
        images = [
            AUO6000Image("a1.bmp", None, "A", 1, 2, result_file="a.csv"),
            AUO6000Image("a2.bmp", None, "A", 2, 2, result_file="a.csv"),
            AUO6000Image("b1.bmp", None, "B", 1, 3, result_file="b.csv"),
            AUO6000Image("b2.bmp", None, "B", 2, 3, result_file="b.csv"),
            AUO6000Image("b3.bmp", None, "B", 3, 3, result_file="b.csv"),
        ]
        dataset = AUO6000Dataset(root="", picture_dir="", images=images, panels=panels)
        self.assertEqual(dataset.expected_cut_points, 0)


if __name__ == "__main__":
    unittest.main()
