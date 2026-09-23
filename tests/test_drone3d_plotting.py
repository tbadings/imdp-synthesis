"""Geometry/camera checks that do not require an OpenGL context."""

from types import SimpleNamespace
import unittest

import numpy as np

from core.plotting.drone3d import REFERENCE_VIEW, _camera_scale, _camera_vectors, _scene, _tiled_box


class TestDrone3DPlotting(unittest.TestCase):
    def test_camera_matches_visvis_rotation(self):
        el, az = np.deg2rad([REFERENCE_VIEW['elevation'], REFERENCE_VIEW['azimuth']])
        a = 3 * np.pi / 2 + el
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        rz = np.array([[np.cos(az), np.sin(az), 0], [-np.sin(az), np.cos(az), 0], [0, 0, 1]])
        eye, up = _camera_vectors(**REFERENCE_VIEW)
        np.testing.assert_allclose(rx @ rz @ eye, [0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(rx @ rz @ up, [0, 1, 0], atol=1e-14)

    def test_domain_corners_fit_camera(self):
        low, high = np.array([-17, -9, -7]), np.array([17, 9, 7])
        corners = np.array(np.meshgrid(*zip(low, high))).reshape(3, -1).T
        eye, up = _camera_vectors(**REFERENCE_VIEW)
        right = np.cross(up, eye)
        for aspect in (0.8, 1000 / 900, 2):
            scale = _camera_scale(low, high, aspect)
            self.assertLess(np.max(np.abs(corners @ up)), scale)
            self.assertLess(np.max(np.abs(corners @ right)), scale * aspect)

    def test_tiles_cover_box_with_outward_normals(self):
        low, high = np.array([-3, 2, -1]), np.array([2, 5, 6])
        vertices, faces, colors = _tiled_box(low, high, [1, 0, 0])
        quads = vertices[faces]
        normals = np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0])
        self.assertTrue(np.all(np.sum(normals * (quads.mean(axis=1) - (low + high) / 2), axis=1) > 0))
        expected_area = 2 * (5 * 3 + 5 * 7 + 3 * 7)
        self.assertAlmostEqual(float(np.linalg.norm(normals, axis=1).sum()), expected_area, places=4)
        np.testing.assert_array_equal(vertices.min(axis=0), low)
        np.testing.assert_array_equal(vertices.max(axis=0), high)
        self.assertEqual(len(colors), len(faces))

    def test_projection_trace_limit_and_repeated_points(self):
        partition = SimpleNamespace(boundary_lb=np.arange(6), boundary_ub=np.arange(6) + 10)
        model = SimpleNamespace(goal=[np.array([np.arange(6), np.arange(6) + 1])], critical=[])
        states = np.array([[1, 90, 2, 80, 3, 70], [1, 91, 2, 81, 3, 71], [4, 92, 5, 82, 6, 72]])
        traces = {0: {'x': states}, 1: {'x': states}}
        low, high, boxes, paths = _scene(partition, model, [0, 2, 4], traces, 1)
        np.testing.assert_array_equal(low, [0, 2, 4])
        np.testing.assert_array_equal(high, [10, 12, 14])
        np.testing.assert_array_equal(boxes[0][0], [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(len(paths), 1)
        np.testing.assert_array_equal(paths[0], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(_scene(partition, model, [0, 2, 4], traces, 0)[3], [])


if __name__ == '__main__':
    unittest.main()
