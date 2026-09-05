from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from router_vision.machine import attach_pictures, load_runs


MODERN_HEADER = (
    "SN,Barcode,Recipe_Name,ProductId,BitDiameter,CuttingSpeed,"
    "OffsetY,OffsetX,Result,BitShiftCount,CuttingTime\n"
)

LEGACY_HEADER = (
    "SN,ID,Barcode,ProductId,CutedTable,Recipe_Name,Result,OffsetX,OffsetY,"
    "RotateAngle,Start_time,Tact_time,OperatorID,Length,Width,ConveyorSpeed,"
    "CuttingTime,Message,MachineID,SubBoardCount,End_time,FixtureBarcode\n"
)


class MachineRunTests(unittest.TestCase):
    def test_modern_result_loads_product_only_and_assigns_picture_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "_20260903_100400.csv").write_text(
                MODERN_HEADER
                + "SN-100,,PRODUCT-A.rcp,PRODUCT-A,1.3,4,-0.02,0.03,True,0,180\n",
                encoding="utf-8",
            )

            run = load_runs(root, "20260903")[0]
            pictures = [
                str(root / "20260903_100029.bmp"),
                str(root / "20260903_100030.bmp"),
                str(root / "20260903_100300.bmp"),
                str(root / "20260903_100400.bmp"),
                str(root / "20260903_100401.bmp"),
            ]
            unmatched = attach_pictures([run], pictures)

            self.assertEqual(run.key, "PRODUCT-A")
            self.assertEqual(run.table, "")
            self.assertEqual(run.start, datetime(2026, 9, 3, 10, 0, 30))
            self.assertEqual(run.end, datetime(2026, 9, 3, 10, 4, 0))
            self.assertEqual(
                [Path(path).name for path in run.pictures],
                [
                    "20260903_100030.bmp",
                    "20260903_100300.bmp",
                    "20260903_100400.bmp",
                ],
            )
            self.assertEqual(unmatched, 2)

    def test_legacy_result_retains_product_and_table_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "_20260903_100015.csv").write_text(
                LEGACY_HEADER
                + "SN-LEGACY,run-1,,PRODUCT-A,LeftTable,PRODUCT-A.rcp,True,"
                "0.1,-0.2,0.03,2026/09/03 10:00:00,15,OP,1,2,3,14,,"
                "AUO6000,1,2026/09/03 10:00:15,FIX\n",
                encoding="utf-8",
            )

            runs = load_runs(root, "20260903")

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].product_id, "PRODUCT-A")
            self.assertEqual(runs[0].table, "LeftTable")
            self.assertEqual(runs[0].key, "PRODUCT-A|LeftTable")
            self.assertEqual(runs[0].start, datetime(2026, 9, 3, 10, 0, 0))
            self.assertEqual(runs[0].end, datetime(2026, 9, 3, 10, 0, 15))

    def test_quoted_commas_are_parsed_as_field_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "_20260903_100200.csv").write_text(
                MODERN_HEADER
                + '"SN,001",,"Recipe, Revision A.rcp",PRODUCT-Q,1.3,4,0,0,'
                "True,0,60\n",
                encoding="utf-8",
            )

            runs = load_runs(root, "20260903")

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].sn, "SN,001")
            self.assertEqual(runs[0].recipe, "Recipe, Revision A.rcp")
            self.assertEqual(runs[0].product_id, "PRODUCT-Q")

    def test_result_with_empty_product_id_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "_20260903_100200.csv").write_text(
                MODERN_HEADER
                + "SN-EMPTY,,UNKNOWN.rcp,,1.3,4,0,0,True,0,60\n",
                encoding="utf-8",
            )

            self.assertEqual(load_runs(root, "20260903"), [])


if __name__ == "__main__":
    unittest.main()
