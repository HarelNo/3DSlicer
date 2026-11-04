"""Tests for nearest-voxel projection output and intersection detection."""

import unittest
from math import ceil

import numpy as np

import main


def reference_projection(volume, points):
    a = np.asarray(points[0], dtype=np.float64)
    b = np.asarray(points[1], dtype=np.float64)
    c = np.asarray(points[2], dtype=np.float64)
    if len(points) == 3:
        row_start = a
        row_start_vector = b - a
        opposite_start = a
        opposite_start_vector = c - a
        row_extent = np.linalg.norm(b - a)
        column_extent = np.linalg.norm(c - b)
    else:
        d = np.asarray(points[3], dtype=np.float64)
        row_start = a
        row_start_vector = d - a
        opposite_start = b
        opposite_start_vector = c - b
        row_extent = max(np.linalg.norm(d - a), np.linalg.norm(c - b))
        column_extent = max(np.linalg.norm(b - a), np.linalg.norm(c - d))

    rows = ceil(row_extent) + 1
    columns = ceil(column_extent) + 1
    image = [[0 for _ in range(columns)] for _ in range(rows)]
    intersects = False
    depth, height, width = volume.shape
    for row in range(rows):
        fraction = row / (rows - 1)
        start = row_start + fraction * row_start_vector
        end = opposite_start + fraction * opposite_start_vector
        span = end - start
        row_length = np.linalg.norm(span)
        x_offset = ((columns - 1) - row_length) / 2
        for column in range(columns):
            if row_length == 0:
                if column != columns // 2:
                    continue
                along_row = 0
            else:
                along_row = (column - x_offset) / row_length
                if not 0 <= along_row <= 1:
                    continue
            x, y, z = start + along_row * span
            if not (
                -1e-8 <= x <= width - 1 + 1e-8
                and -1e-8 <= y <= height - 1 + 1e-8
                and -1e-8 <= z <= depth - 1 + 1e-8
            ):
                continue
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            z = min(max(z, 0), depth - 1)
            image[row][column] = volume[
                int(np.floor(z + 0.5)),
                int(np.floor(y + 0.5)),
                int(np.floor(x + 0.5)),
            ]
            intersects = True
    return image, intersects


class ProjectCoordinatesTests(unittest.TestCase):
    def test_vectorized_triangle_matches_nearest_neighbor_reference(self):
        volume = np.arange(9 * 10 * 11, dtype=np.uint16).reshape(9, 10, 11) % 256
        volume = volume.astype(np.uint8)
        points = ((4.25, 3.5, 3), (0.25, 7.5, 5), (8.25, 7.5, 5))

        expected = reference_projection(volume, points)
        actual = main._project_coordinates_with_intersection(volume, points)

        self.assertEqual(actual, expected)

    def test_vectorized_trapezoid_matches_nearest_neighbor_reference(self):
        volume = np.arange(9 * 10 * 11, dtype=np.uint16).reshape(9, 10, 11) % 256
        volume = volume.astype(np.uint8)
        points = ((-1, 2, 1), (5, 2, 1), (4.5, 8, 7), (-0.5, 8, 7))

        expected = reference_projection(volume, points)
        actual = main._project_coordinates_with_intersection(volume, points)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
