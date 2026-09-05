from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from production_app import (
    APP_FULL_NAME,
    APP_NAME,
    MIN_TRAINING_IMAGES_PER_CLASS,
    RECENT_RESULT_LIMIT,
    SETTINGS_PASSWORD,
    STATE_TEXT,
    model_file_name,
    render_prediction_overview,
)
from router_vision.interlock import GateState
from router_vision.model import CropBox
from router_vision.production import ImageDecision, PanelDecision


class RecordingExtractor:
    def __init__(self):
        self.paths: list[str] = []

    def edge_map(self, path, crop):
        self.paths.append(str(path))
        edge = np.zeros((40, 80), dtype=np.uint8)
        cv2.rectangle(edge, (12, 5), (68, 35), 255, 2)
        return edge


class OverviewClassifier:
    def __init__(self):
        self.crop = CropBox(0, 0, 80, 40)
        self.extractor = RecordingExtractor()


def fake_run():
    return SimpleNamespace(sn="PANEL-001", run_id="RUN-1", result_file="run.csv")


class ProductionOverviewTests(unittest.TestCase):
    def test_auo6000_training_model_is_keyed_by_product_id(self):
        self.assertEqual(MIN_TRAINING_IMAGES_PER_CLASS, 5)
        self.assertEqual(
            model_file_name("199944000-Optoput-Test"),
            "cut_classifier_199944000-Optoput-Test.pt",
        )

    def test_overview_contains_every_original_and_sobel_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            details = []
            for index in range(4):
                path = Path(folder) / f"cut-{index}.bmp"
                original = np.full((40, 80, 3), (30 + index * 20, 90, 140),
                                   dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), original))
                label = "NG" if index == 2 else "GOOD"
                confidence = 0.91 if label == "NG" else 0.99
                details.append(ImageDecision(index, str(path), label, confidence))

            result = PanelDecision(
                run=fake_run(), status="NG", note="cut point 3 is NG",
                checked=4, details=details,
            )
            classifier = OverviewClassifier()
            overview = render_prediction_overview(
                classifier, result, 0.95, cell_width=260, cell_height=180)

            # Four cells form a 2x2 tile-only overview with fixed gaps. The Qt
            # view does not add another title or panel-summary row above it.
            self.assertEqual(overview.shape, (390, 550, 3))
            self.assertGreater(int(overview.max()), 0)
            self.assertEqual(classifier.extractor.paths,
                             [detail.path for detail in details])
            # Prediction text stays in the header and no longer obscures the
            # centre of the original-image evidence pane.
            self.assertEqual(tuple(overview[103, 74]), (30, 90, 140))

    def test_empty_result_still_returns_an_operator_message_image(self):
        classifier = OverviewClassifier()
        result = PanelDecision(run=fake_run(), status="FAULT")
        overview = render_prediction_overview(classifier, result)
        self.assertEqual(overview.shape, (240, 720, 3))
        self.assertGreater(int(overview.max()), 0)

    def test_production_ui_source_contains_no_thai_text(self):
        source = (Path(__file__).parents[1] / "production_app.py").read_text(
            encoding="utf-8")
        thai = [character for character in source if "\u0e00" <= character <= "\u0e7f"]
        self.assertEqual(thai, [])

    def test_avtr_brand_and_recent_board_limit(self):
        source = (Path(__file__).parents[1] / "production_app.py").read_text(
            encoding="utf-8")
        self.assertEqual(APP_NAME, "AVTR")
        self.assertEqual(APP_FULL_NAME, "Automatic Vision Tab Router")
        self.assertEqual(RECENT_RESULT_LIMIT, 5)
        self.assertEqual(STATE_TEXT[GateState.PASS], "GOOD")
        self.assertEqual(STATE_TEXT[GateState.NG], "NG")
        self.assertEqual(STATE_TEXT[GateState.FAULT], "NG")
        self.assertNotIn("RELEASE / HOLD REASON", source)
        self.assertNotIn("lbl_state_reason", source)
        self.assertNotIn("lbl_release_big", source)
        self.assertNotIn("setBackground(", source)
        self.assertNotIn("content_layout.addLayout(cards)", source)
        self.assertNotIn("lbl_panel", source)
        self.assertNotIn("AVTR | ALL CUT POINTS", source)
        self.assertNotIn("cut points displayed", source)
        self.assertNotIn("image_layout.addWidget(self.lbl_detail)", source)
        self.assertIn(
            "self.table.cellClicked.connect(self._open_recent_result)", source
        )
        self.assertIn("class ResultImageDialog", source)
        self.assertIn(
            'os.environ.get("AVTR_SETTINGS_PASSWORD", "").strip()', source
        )
        self.assertIn('QPushButton("SETTING ▾")', source)
        self.assertIn("QLineEdit.Password", source)
        self.assertIn("QLocale.setDefault", source)
        self.assertIn("def _preview_sobel", source)
        self.assertIn("def _test_vision_transformer", source)
        self.assertIn("spin_confidence", source)
        self.assertIn('QLabel("1. AUROTEK AUO6000 DATA PATH")', source)
        self.assertIn(
            "Select one machine export root containing Picture, Result, Log and Recipe.",
            source,
        )
        self.assertIn("QFileDialog.getExistingDirectory", source)
        self.assertIn("def _set_settings_path", source)
        self.assertIn("scan_auo6000_dataset", source)
        self.assertIn("self.settings_image_paths", source)
        self.assertIn('QPushButton("SELECT AUO6000 PATH")', source)
        self.assertIn("ROUTER RESULT", source)
        self.assertIn("CUT POINT", source)
        self.assertIn("SOBEL: NOT SAVED", source)
        self.assertIn("TRAINING: NOT TRAINED", source)
        self.assertIn("def _save_current_sobel", source)
        self.assertIn("sobel_workflow.json", source)
        self.assertIn("NEXT NOT SAVED", source)
        self.assertIn('QLabel("2. FINE-TUNE SOBEL EDGE DETECTION")', source)
        self.assertIn("QSlider(Qt.Horizontal)", source)
        self.assertIn('"Blur strength"', source)
        self.assertIn('"X edge weight"', source)
        self.assertIn('"Y edge weight"', source)
        self.assertIn('"Edge brightness"', source)
        self.assertIn('"Noise removal"', source)
        self.assertIn("lbl_workflow_progress", source)
        self.assertIn("QSplitter(Qt.Horizontal)", source)
        self.assertIn("LIVE ORIGINAL / SOBEL COMPARISON", source)
        self.assertIn("SOBEL PARAMETERS", source)
        self.assertIn("Decimal controls use precise 0.01 steps.", source)
        self.assertIn("self.slider_sobel_clip.setRange(9000, 10000)", source)
        self.assertIn('QPushButton("RESET PARAMETERS")', source)
        self.assertIn('QPushButton("◀")', source)
        self.assertIn('QPushButton("▶")', source)
        self.assertIn("self.sobel_preview_timer.start()", source)
        self.assertNotIn('QPushButton("PREVIEW SOBEL")', source)
        self.assertIn("3. IMAGE CLASSIFICATION BY VISION TRANSFORMER", source)
        self.assertIn('QTabWidget()', source)
        self.assertIn("tabBar().setExpanding(True)", source)
        self.assertIn('"1  DATA SOURCE"', source)
        self.assertIn('"2  SOBEL TUNING"', source)
        self.assertIn('"3  VISION TRANSFORMER"', source)
        self.assertIn("SOURCE FOLDERS", source)
        self.assertIn("tbl_auo_folders", source)
        self.assertIn("CONTINUE TO SOBEL TUNING", source)
        self.assertIn("READY FOR SOBEL TUNING", source)
        self.assertIn("_load_sobel_source_path", source)
        self.assertNotIn("Setting Notice :", source)
        self.assertLess(
            source.index("image_navigation.addWidget(self.cmb_settings_image, 1)"),
            source.index("image_navigation.addWidget(self.btn_previous_image)"),
        )
        self.assertLess(
            source.index("image_navigation.addWidget(self.btn_previous_image)"),
            source.index("image_navigation.addWidget(self.btn_next_image)"),
        )
        self.assertNotIn("Follow steps 1–3", source)
        self.assertNotIn("ADVANCED OPTIONS ▾", source)
        self.assertNotIn("_toggle_advanced_settings", source)
        self.assertIn('setObjectName("settingsPanel")', source)
        self.assertIn('setObjectName("settingsCard")', source)
        self.assertIn('setObjectName("testAction")', source)
        self.assertIn('setObjectName("saveAction")', source)
        self.assertIn("1. LABEL SOBEL IMAGES", source)
        self.assertIn("2. TRAIN MODEL", source)
        self.assertIn("3. TEST MODEL", source)
        self.assertIn('QPushButton("LABEL GOOD")', source)
        self.assertIn('QPushButton("LABEL NG")', source)
        self.assertIn('QPushButton("TRAIN & SAVE MODEL")', source)
        self.assertIn("MIN_TRAINING_IMAGES_PER_CLASS = 5", source)
        self.assertIn("class ModelTrainingWorker", source)
        self.assertIn("def _train_settings_model", source)
        self.assertIn("model_file_name(product)", source)
        self.assertIn(
            "Testing uses the Sobel preprocessing saved in the model.", source
        )
        self.assertIn("self.settings_classifier", source)
        self.assertIn("def _load_settings_model", source)
        self.assertIn("No trained Sobel ViT model for ProductId", source)
        self.assertNotIn('QPushButton("OPEN LARGE VIEW")', source)
        self.assertIn("IMAGE REVIEW / PREDICTION RESULTS", source)
        self.assertIn("vision_workspace = QSplitter(Qt.Horizontal)", source)
        self.assertIn("def _layout_sobel_parameter_cards", source)
        self.assertIn("setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)", source)
        self.assertIn("self.trial_preview_pixmap", source)
        self.assertIn("MANUAL LABEL", source)
        self.assertIn('prediction_text = f"PREDICT:', source)
        self.assertIn('active_separator = QLabel("|")', source)
        self.assertIn('f"Recipe Name : {spec.get', source)
        self.assertNotIn('f"ACTIVE:', source)
        self.assertLess(
            source.index("control_layout.addWidget(active_separator)"),
            source.index("control_layout.addWidget(self.lbl_active_product)"),
        )
        self.assertNotIn("ENGINEER ACCESS", source)
        self.assertNotIn("MetricCard", source)
        self.assertNotIn("card_result", source)
        self.assertNotIn("card_cycle", source)
        self.assertNotIn("card_pass", source)
        self.assertNotIn("card_ng", source)
        self.assertNotIn("card_queue", source)


if __name__ == "__main__":
    unittest.main()
