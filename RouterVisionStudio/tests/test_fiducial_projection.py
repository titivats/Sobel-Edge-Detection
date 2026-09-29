import math
import unittest

import numpy as np
from router_vision.fiducial_projection import align_two_fiducials, project_aligned_edge


class FiducialProjectionTests(unittest.TestCase):
    def project(self, **changes):
        settings = dict(
            rotation_rad=0,
            translation_mm=[0, 0],
            material_side=1,
            tool_radius_mm=0.5,
            camera_center_mm=[0, 0],
            camera_center_px=[100, 100],
            camera_x_angle_rad=0,
            scale_mm_per_px=[0.1, 0.1],
            image_y_up=True,
        )
        settings.update(changes)
        return project_aligned_edge([[0, 0], [2, 0]], **settings)

    def test_two_fiducials_rotate_then_translate_the_selected_edge(self):
        fit = align_two_fiducials([[0, 0], [2, 0]], [[10, 20], [10, 22]])
        self.assertAlmostEqual(fit["rotation_rad"], math.pi / 2)
        self.assertLess(fit["max_fiducial_residual_mm"], 1e-10)
        result = self.project(
            rotation_rad=fit["rotation_rad"],
            translation_mm=fit["translation_mm"],
            camera_center_mm=[10, 20],
        )
        np.testing.assert_allclose(result["aligned_edge_mm"], [[9.5, 20], [9.5, 22]])
        np.testing.assert_allclose(result["edge_px"], [[95, 100], [95, 80]])

    def test_fixed_camera_preserves_fiducial_translation_in_pixels(self):
        result = self.project(translation_mm=[1, 2])
        np.testing.assert_allclose(result["edge_px"], [[110, 75], [130, 75]])

    def test_camera_following_same_translation_cancels_only_translation(self):
        result = self.project(translation_mm=[1, 2], camera_center_mm=[1, 2])
        np.testing.assert_allclose(result["edge_px"], self.project()["edge_px"])

    def test_independent_camera_error_does_not_cancel(self):
        result = self.project(translation_mm=[1, 2], camera_center_mm=[1.1, 2])
        np.testing.assert_allclose(result["edge_px"], [[99, 95], [119, 95]])

    def test_material_side_is_applied_before_alignment(self):
        result = self.project(material_side=-1, rotation_rad=math.pi / 2)
        np.testing.assert_allclose(result["aligned_edge_mm"], [[0.5, 0], [0.5, 2]], atol=1e-12)

    def test_spacing_disagreement_is_reported_without_scaling_path(self):
        fit = align_two_fiducials([[0, 0], [2, 0]], [[10, 0], [13, 0]])
        self.assertAlmostEqual(fit["fiducial_spacing_difference_mm"], 1)
        self.assertAlmostEqual(fit["max_fiducial_residual_mm"], 0.5)

    def test_camera_rotation_and_anisotropic_scale(self):
        result = self.project(camera_x_angle_rad=math.pi / 2, scale_mm_per_px=[0.1, 0.2])
        np.testing.assert_allclose(result["edge_px"], [[105, 100], [105, 110]])

    def test_missing_pose_invalid_side_or_degenerate_fiducials_rejected(self):
        for change in (
            {"camera_center_mm": None},
            {"material_side": 0},
            {"scale_mm_per_px": [0, 0.1]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.project(**change)
        with self.assertRaises(ValueError):
            align_two_fiducials([[0, 0], [0, 0]], [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
