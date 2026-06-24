from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from realtime_overlay import status_from_classes
from realtime_product_info import (
    CsvPointCounter,
    ProductCsvIndex,
    ProductInfoCache,
    find_product_csv,
    read_product_info,
    timestamp_from_name,
)


class RealtimePredictUiTest(unittest.TestCase):
    def test_timestamp_from_name_reads_picture_and_csv_names(self) -> None:
        self.assertEqual(
            timestamp_from_name(Path("20260619_082849.bmp")).strftime("%Y-%m-%d %H:%M:%S"),
            "2026-06-19 08:28:49",
        )
        self.assertEqual(
            timestamp_from_name(Path("_20260619_082355.csv")).strftime("%Y-%m-%d %H:%M:%S"),
            "2026-06-19 08:23:55",
        )

    def test_find_product_csv_uses_latest_csv_before_image_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir)
            (csv_dir / "_20260619_082355.csv").write_text("Time\n", encoding="utf-8")
            expected = csv_dir / "_20260619_082757.csv"
            expected.write_text("Time\n", encoding="utf-8")
            (csv_dir / "_20260619_083019.csv").write_text("Time\n", encoding="utf-8")

            csv_path = find_product_csv(Path("20260619_082849.bmp"), csv_dir)

        self.assertEqual(csv_path.name, expected.name)

    def test_read_product_info_reads_first_data_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "_20260619_082355.csv"
            csv_path.write_text(
                "SN,ProductId,CutedTable,Recipe_Name,Result,End_time\n"
                "31733,160914002C01,RightTable,160914002C01.rcp,True,2026/06/19 08:23:55\n",
                encoding="utf-8",
            )

            info = read_product_info(csv_path)

        self.assertEqual(info.sn, "31733")
        self.assertEqual(info.cuted_table, "RightTable")
        self.assertEqual(info.product_id, "160914002C01")
        self.assertEqual(info.recipe_name, "160914002C01.rcp")
        self.assertEqual(info.result, "True")

    def test_status_from_classes(self) -> None:
        self.assertEqual(status_from_classes(["In spec"]), "PASS")
        self.assertEqual(status_from_classes(["In spec", "Out spec"]), "NG")
        self.assertEqual(status_from_classes(["PASS"]), "PASS")
        self.assertEqual(status_from_classes(["PASS", "NG"]), "NG")
        self.assertEqual(status_from_classes(["unknown"]), "UNKNOWN")
        self.assertEqual(status_from_classes([]), "UNKNOWN")

    def test_csv_point_counter_resets_per_csv_file(self) -> None:
        counter = CsvPointCounter()

        self.assertEqual(counter.next_point(Path("_20260619_082757.csv")), 1)
        self.assertEqual(counter.next_point(Path("_20260619_082757.csv")), 2)
        self.assertEqual(counter.next_point(Path("_20260619_083019.csv")), 1)

    def test_product_csv_index_matches_without_rebuilding_each_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir)
            (csv_dir / "_20260619_082355.csv").write_text("Time\n", encoding="utf-8")
            expected = csv_dir / "_20260619_082757.csv"
            expected.write_text("Time\n", encoding="utf-8")
            index = ProductCsvIndex(csv_dir)

            self.assertEqual(index.match(Path("20260619_082849.bmp")), expected)
            signature = index.signature
            self.assertEqual(index.match(Path("20260619_082904.bmp")), expected)

        self.assertEqual(index.signature, signature)

    def test_product_info_cache_reuses_csv_until_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "_20260619_082355.csv"
            csv_path.write_text("SN,CutedTable\n31733,RightTable\n", encoding="utf-8")
            cache = ProductInfoCache()

            first = cache.read(csv_path)
            second = cache.read(csv_path)

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
