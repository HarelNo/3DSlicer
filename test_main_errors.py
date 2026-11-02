"""Tests that invalid inputs to main.py raise useful errors."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main


class LoadNrrdGrayscaleErrorTests(unittest.TestCase):
    # Checks that the loader rejects NRRD data that is not three-dimensional.
    def test_rejects_non_3d_nrrd_data(self):
        data = np.zeros((2, 2), dtype=np.int16)
        with patch.object(main.nrrd, "read", return_value=(data, {"dimension": 2})):
            with self.assertRaisesRegex(ValueError, "must contain a 3D"):
                main.load_nrrd_grayscale("invalid.nrrd")

    # Checks that the loader rejects non-numeric voxel values.
    def test_rejects_non_numeric_nrrd_data(self):
        data = np.full((2, 2, 2), "not a number")
        with patch.object(main.nrrd, "read", return_value=(data, {"dimension": 3})):
            with self.assertRaisesRegex(ValueError, "numeric scalar"):
                main.load_nrrd_grayscale("invalid.nrrd")

    # Checks that the loader rejects NaN or infinite voxel values.
    def test_rejects_non_finite_nrrd_data(self):
        data = np.full((2, 2, 2), np.nan)
        with patch.object(main.nrrd, "read", return_value=(data, {"dimension": 3})):
            with self.assertRaisesRegex(ValueError, "finite values"):
                main.load_nrrd_grayscale("invalid.nrrd")

    # Checks that the loader rejects a volume with an empty dimension.
    def test_rejects_empty_nrrd_data(self):
        data = np.empty((0, 2, 2), dtype=np.int16)
        with patch.object(main.nrrd, "read", return_value=(data, {"dimension": 3})):
            with self.assertRaises(ValueError):
                main.load_nrrd_grayscale("empty.nrrd")


class ProjectCoordinatesErrorTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.zeros((4, 4, 4), dtype=np.uint8)
        self.valid_triangle = ((1, 1, 1), (1, 3, 1), (3, 1, 1))

    # Checks that projection rejects invalid shapes, types, or voxel values.
    def test_rejects_invalid_matrices(self):
        invalid_matrices = (
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((0, 4, 4), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=float),
            np.full((4, 4, 4), 256, dtype=np.int16),
        )
        for matrix in invalid_matrices:
            with self.subTest(shape=matrix.shape, dtype=matrix.dtype):
                with self.assertRaises(ValueError):
                    main.project_coordinates(matrix, self.valid_triangle)

    # Checks that projection requires exactly three or four defining points.
    def test_rejects_wrong_number_of_points(self):
        with self.assertRaisesRegex(ValueError, "3 or 4 points"):
            main.project_coordinates(self.matrix, ((0, 0, 0), (1, 1, 1)))

    # Checks that projection rejects malformed and non-finite coordinates.
    def test_rejects_malformed_or_non_finite_points(self):
        invalid_triangles = (
            ((1, 1), (1, 3, 1), (3, 1, 1)),
            ((1, 1, 1), (1, 3, 1), (3, 1, np.inf)),
            ((1, 1, 1), ("bad", 3, 1), (3, 1, 1)),
        )
        for points in invalid_triangles:
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    main.project_coordinates(self.matrix, points)

    # Checks that a triangle with repeated vertices is rejected.
    def test_rejects_repeated_points(self):
        points = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            main.project_coordinates(self.matrix, points)

    # Checks that a triangle whose apex is not equidistant from its base is rejected.
    def test_rejects_non_isosceles_triangle(self):
        points = ((1, 1, 1), (1, 3, 1), (3, 2, 1))
        with self.assertRaisesRegex(ValueError, "must be isosceles"):
            main.project_coordinates(self.matrix, points)

    # Checks that collinear triangle points are rejected as a degenerate plane.
    def test_rejects_collinear_triangle(self):
        points = ((1, 1, 1), (1, 2, 1), (1, 0, 1))
        with self.assertRaisesRegex(ValueError, "non-degenerate plane"):
            main.project_coordinates(self.matrix, points)


class LoadCoordinateSeriesErrorTests(unittest.TestCase):
    def assert_coordinate_file_raises(self, contents, message):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "coordinates.txt"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                main.load_coordinate_series(path)

    # Checks that coordinate rows must contain exactly three comma-separated values.
    def test_rejects_rows_with_wrong_number_of_fields(self):
        self.assert_coordinate_file_raises("1, 2\n", "must contain x, y, z")

    # Checks that coordinate rows containing non-numeric values are rejected.
    def test_rejects_non_numeric_coordinates(self):
        self.assert_coordinate_file_raises(
            "1, bad, 3\n2, 3, 4\n3, 4, 5\n",
            "non-numeric coordinate",
        )

    # Checks that coordinate rows containing NaN or infinity are rejected.
    def test_rejects_non_finite_coordinates(self):
        self.assert_coordinate_file_raises(
            "1, 2, nan\n2, 3, 4\n3, 4, 5\n",
            "must be finite",
        )

    # Checks that each coordinate group must contain three or four points.
    def test_rejects_series_with_wrong_number_of_points(self):
        self.assert_coordinate_file_raises("1, 2, 3\n2, 3, 4\n", "exactly 3 or 4")

    # Checks that a coordinate file with no groups is rejected.
    def test_rejects_empty_coordinate_file(self):
        self.assert_coordinate_file_raises("", "no coordinate series found")


if __name__ == "__main__":
    unittest.main()
