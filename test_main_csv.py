"""Tests for loading coordinate series from CSV files."""

import tempfile
import unittest
from pathlib import Path

import main
import numpy as np


class LoadCoordinateCsvTests(unittest.TestCase):
    def load_csv(self, contents):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "coordinates.csv"
        path.write_text(contents, encoding="utf-8")
        return main.load_coordinate_series(path)

    def test_reads_header_and_triangle_and_trapezoid_rows(self):
        series = self.load_csv(
            "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,"
            "x3,y3,z3,x4,y4,z4\n"
            "3,1,2,3,4,5,6,7,8,9,,,\n"
            "4,1,2,3,4,5,6,7,8,9,10,11,12\n"
        )

        self.assertEqual(
            series,
            [
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
                (
                    (1.0, 2.0, 3.0),
                    (4.0, 5.0, 6.0),
                    (10.0, 11.0, 12.0),
                    (7.0, 8.0, 9.0),
                ),
            ],
        )

    def test_csv_trapezoid_parallel_side_pairs_project_without_pinching(self):
        series = self.load_csv(
            "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,"
            "x3,y3,z3,x4,y4,z4\n"
            "4,0,0,0,0,4,0,4,0,0,4,4,0\n"
        )
        volume = np.full((2, 5, 5), 255, dtype=np.uint8)

        image = np.asarray(main.project_coordinates(volume, series[0]))

        self.assertEqual(image.shape, (5, 5))
        np.testing.assert_array_equal(image, np.full((5, 5), 255, dtype=np.uint8))

    def test_skips_comment_lines_and_reads_repository_csv_example(self):
        path = Path(__file__).parent / "csv files" / "lecturer_example.csv"

        series = main.load_coordinate_series(path)

        self.assertEqual(len(series), 4)
        self.assertEqual(tuple(map(len, series)), (3, 4, 3, 4))

    def test_rejects_missing_coordinate_fields(self):
        with self.assertRaisesRegex(ValueError, "complete xyz points"):
            self.load_csv("3,1,2,3,4,5,6,7,8\n")

    def test_rejects_non_numeric_coordinate_fields(self):
        with self.assertRaisesRegex(ValueError, "non-numeric coordinate"):
            self.load_csv("3,1,2,3,4,5,6,7,bad,9\n")


if __name__ == "__main__":
    unittest.main()
